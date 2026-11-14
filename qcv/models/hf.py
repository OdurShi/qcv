"""Teacher-forced complete-alphabet scoring with a frozen Hugging Face VLM (Eq. 2).

For a question ``q_k`` and each canonical answer ``a`` of its alphabet, the
scorer feeds ``prompt(q_k) + a`` to the model once and returns

    l_k(a) = (1 / L(a)) * sum_t log p(a_t | I, q_k, a_<t),

the mean log-probability of the ``L(a)`` answer tokens. Special, padding and
end-of-answer tokens are not scored. All candidates of one question are scored
in one padded batch (optionally split into chunks to bound memory).

Not executed at paper scale in this repository's validation; the pure scoring
logic is covered by tests with a stub processor and model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..types import QuestionSpec
from .media import load_media

ANSWER_SEPARATORS = {"none": "", "space": " "}
PROMPT_TEMPLATE = (
    "{question}\nReturn exactly one canonical answer from this finite set: [{alphabet}].\nAnswer:"
)


@dataclass(frozen=True)
class ModelLoadOptions:
    model_name_or_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attention_implementation: str | None = None
    max_batch_candidates: int = 0
    """0 scores the whole alphabet in one batch; a positive value is the chunk size."""
    answer_separator: str = "auto"
    """``auto``: no separator after a chat template, one space after a plain prompt."""

    def __post_init__(self) -> None:
        if self.max_batch_candidates < 0:
            raise ValueError("max_batch_candidates must be zero or a positive integer")
        if self.answer_separator not in {"auto", *ANSWER_SEPARATORS}:
            raise ValueError(f"Unsupported answer_separator {self.answer_separator!r}")


def build_instruction(question: QuestionSpec) -> str:
    """User-turn text: the question followed by its finite answer alphabet."""
    return PROMPT_TEMPLATE.format(
        question=question.text, alphabet=", ".join(question.role.alphabet)
    )


def find_last_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
    """Start index of the last occurrence of ``subsequence`` in ``sequence``."""
    width = len(subsequence)
    if width == 0 or width > len(sequence):
        return None
    target = list(subsequence)
    for start in range(len(sequence) - width, -1, -1):
        if list(sequence[start : start + width]) == target:
            return start
    return None


def answer_token_positions(
    input_ids: Sequence[int], attention_mask: Sequence[int], answer_ids: Sequence[int]
) -> list[int]:
    """Positions in the padded row of the answer tokens (its last occurrence).

    Works for left- and right-padded rows: the search runs over the non-padding
    tokens only and maps the match back to padded positions.
    """
    real_positions = [index for index, flag in enumerate(attention_mask) if flag]
    real_ids = [int(input_ids[index]) for index in real_positions]
    start = find_last_subsequence(real_ids, [int(token) for token in answer_ids])
    if start is None:
        raise RuntimeError("Could not locate the answer tokens in the model input")
    positions = real_positions[start : start + len(answer_ids)]
    if positions[0] == 0:
        raise RuntimeError("The answer starts at the first position; no causal context exists")
    return positions


def mean_answer_logprob(logits: Any, input_ids: Any, positions: Sequence[int]) -> float:
    """Mean log-probability of the tokens at ``positions`` under next-token ``logits`` (Eq. 2).

    ``logits`` is the ``(sequence, vocabulary)`` output of one row; the token at
    position ``t`` is predicted by the logits at position ``t - 1``. The
    log-softmax is taken only at the scored positions.
    """
    import torch

    index = torch.as_tensor(list(positions), device=logits.device)
    log_probabilities = torch.log_softmax(logits[index - 1].float(), dim=-1)
    tokens = input_ids[index].to(log_probabilities.device)
    return float(log_probabilities.gather(1, tokens[:, None]).mean().item())


class HuggingFaceVLMScorer:
    """Teacher-forced complete-alphabet scorer for Hugging Face image/video-text models."""

    def __init__(self, options: ModelLoadOptions, processor: Any = None, model: Any = None):
        """Load the processor and model, or wrap already constructed ones (used by tests)."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFaceVLMScorer requires torch and transformers: pip install -e '.[vlm]'"
            ) from exc
        self.torch = torch
        self.options = options
        if (processor is None) != (model is None):
            raise ValueError("Pass both processor and model, or neither")
        if processor is None:
            processor, model = self._load(options)
        self.processor = processor
        self.model = model
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def _load(self, options: ModelLoadOptions) -> tuple[Any, Any]:
        import transformers

        torch = self.torch
        dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if options.dtype not in dtypes:
            raise ValueError(f"Unsupported dtype {options.dtype!r}")
        processor = transformers.AutoProcessor.from_pretrained(
            options.model_name_or_path, trust_remote_code=options.trust_remote_code
        )
        # transformers renamed the ``torch_dtype`` argument to ``dtype`` in version 4.56.
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        load_kwargs: dict[str, Any] = {
            "dtype" if version >= (4, 56) else "torch_dtype": dtypes[options.dtype],
            "trust_remote_code": options.trust_remote_code,
        }
        if options.attention_implementation:
            load_kwargs["attn_implementation"] = options.attention_implementation
        errors: list[str] = []
        for loader_name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(options.model_name_or_path, **load_kwargs)
            except Exception as exc:  # each auto class fails differently on a mismatch
                errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
                continue
            return processor, model.eval().to(options.device)
        details = " | ".join(errors) if errors else "no compatible auto-model class"
        raise RuntimeError(f"Could not load {options.model_name_or_path!r}: {details}")

    def _has_chat_template(self) -> bool:
        return bool(
            getattr(self.processor, "chat_template", None)
            or getattr(self.tokenizer, "chat_template", None)
        )

    def build_prompt(self, question: QuestionSpec, media_type: str) -> tuple[str, str]:
        """Return the prompt up to the answer and the separator placed before the answer."""
        instruction = build_instruction(question)
        if self._has_chat_template():
            content: list[dict[str, Any]] = [{"type": media_type}]
            content.append({"type": "text", "text": instruction})
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            automatic = ""
        else:
            prompt = f"<{media_type}>\n{instruction}"
            automatic = " "
        if self.options.answer_separator == "auto":
            return str(prompt), automatic
        return str(prompt), ANSWER_SEPARATORS[self.options.answer_separator]

    def _encode(self, texts: Sequence[str], media_type: str, media_value: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"text": list(texts), "padding": True, "return_tensors": "pt"}
        kwargs["images" if media_type == "image" else "videos"] = [media_value for _ in texts]
        try:
            encoded = self.processor(**kwargs)
        except TypeError:
            if media_type != "video":
                raise
            # Processors without a video modality receive the sampled frames as images.
            kwargs["images"] = kwargs.pop("videos")
            encoded = self.processor(**kwargs)
        return {
            key: value.to(self.options.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def score_question(self, question: QuestionSpec, media: Mapping[str, Any]) -> tuple[float, ...]:
        """Length-normalised log-score ``l_k(a)`` of every answer in the question's alphabet."""
        torch = self.torch
        media_type, media_value = load_media(media)
        prompt, separator = self.build_prompt(question, media_type)
        answers = list(question.role.alphabet)
        batch_size = self.options.max_batch_candidates or len(answers)
        scores: list[float] = []
        for start in range(0, len(answers), batch_size):
            suffixes = [separator + answer for answer in answers[start : start + batch_size]]
            encoded = self._encode(
                [prompt + suffix for suffix in suffixes], media_type, media_value
            )
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise RuntimeError("The processor did not return input_ids")
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            if logits.shape[1] != input_ids.shape[1]:
                raise RuntimeError(
                    "Model logits and input_ids differ in length; use a processor that expands "
                    "media placeholder tokens in input_ids"
                )
            for row, suffix in enumerate(suffixes):
                answer_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
                try:
                    positions = answer_token_positions(
                        input_ids[row].tolist(), attention_mask[row].tolist(), answer_ids
                    )
                except RuntimeError as exc:
                    raise RuntimeError(f"Answer {answers[start + row]!r}: {exc}") from exc
                scores.append(mean_answer_logprob(logits[row], input_ids[row], positions))
        return tuple(scores)
