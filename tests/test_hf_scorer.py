"""Pure-logic tests of the Hugging Face scorer with a stub processor and a stub model.

The stub model is a bigram language model: the logits at position t depend only
on the token at position t. It is therefore causal and blind to padding, and
every expected log-probability can be computed by hand from its logit table.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from qcv.models.hf import (
    PROMPT_TEMPLATE,
    HuggingFaceVLMScorer,
    ModelLoadOptions,
    answer_token_positions,
    build_instruction,
    find_last_subsequence,
    mean_answer_logprob,
)
from qcv.types import QuestionSpec, RoleSpec

torch = pytest.importorskip("torch")

VOCABULARY = 64


class StubTokenizer:
    """Whitespace tokenizer; id 0 is the padding token."""

    def __init__(self):
        self.vocabulary = {"<pad>": 0}

    def encode(self, text):
        return [self.vocabulary.setdefault(token, len(self.vocabulary)) for token in text.split()]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}


class StubProcessor:
    def __init__(self, padding_side="right", chat=False, video_modality=True):
        self.tokenizer = StubTokenizer()
        self.padding_side = padding_side
        self.video_modality = video_modality
        self.calls = []
        if chat:
            self.chat_template = "stub"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"<{item['type']}>" if item["type"] != "text" else item["text"]
                 for item in messages[0]["content"]]  # fmt: skip
        return "<user> " + " ".join(parts) + " <assistant>\n"

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt", **extra):
        if videos is not None and not self.video_modality:
            raise TypeError("unexpected keyword argument 'videos'")
        self.calls.append({"images": images, "videos": videos, "text": list(text)})
        rows = [
            self.tokenizer.encode(item.replace("<image>", "<img> <img> <img>")) for item in text
        ]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            span = (
                slice(width - len(row), width) if self.padding_side == "left" else slice(len(row))
            )
            input_ids[index, span] = torch.tensor(row)
            attention_mask[index, span] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BigramModel:
    def __init__(self, seed=0, drop_last=False):
        generator = torch.Generator().manual_seed(seed)
        self.table = torch.randn((VOCABULARY, VOCABULARY), generator=generator)
        self.drop_last = drop_last

    def __call__(self, input_ids, attention_mask=None, **extra):
        logits = self.table[input_ids]
        return SimpleNamespace(logits=logits[:, :-1] if self.drop_last else logits)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    return {"type": "image", "path": str(path)}


@pytest.fixture
def question():
    role = RoleSpec("colour", ("red", "dark red", "very dark red", "blue"))
    return QuestionSpec("q0", "What colour is the cube?", "colour", role, 0, is_target=True)


def _scorer(processor=None, model=None, **options):
    return HuggingFaceVLMScorer(
        ModelLoadOptions("stub", device="cpu", **options),
        processor=processor or StubProcessor(),
        model=model or BigramModel(),
    )


def _expected(scorer, prompt, answer):
    """Mean over the answer tokens of log p(token | previous token) from the bigram table."""
    encode = scorer.tokenizer.encode
    previous = encode(prompt.replace("<image>", "<img> <img> <img>"))[-1]
    log_probabilities = torch.log_softmax(scorer.model.table, dim=-1)
    values = []
    for token in encode(answer):
        values.append(float(log_probabilities[previous, token]))
        previous = token
    return float(np.mean(values))


def test_scores_are_mean_log_probabilities_of_the_answer_tokens(question, image):
    scorer = _scorer()
    scores = scorer.score_question(question, image)
    prompt, separator = scorer.build_prompt(question, "image")
    assert prompt == "<image>\n" + build_instruction(question) and separator == " "
    assert len(scores) == 4
    for score, answer in zip(scores, question.role.alphabet, strict=True):
        assert score == pytest.approx(_expected(scorer, prompt, answer), abs=1e-5)
    # Length normalisation: the three-token answer is a mean, not a sum, of token log-probs.
    table = torch.log_softmax(scorer.model.table, dim=-1)
    very, dark, red = scorer.tokenizer.encode("very dark red")
    last_prompt_token = scorer.tokenizer.encode(prompt.replace("<image>", "<img>"))[-1]
    total = table[last_prompt_token, very] + table[very, dark] + table[dark, red]
    assert scores[2] == pytest.approx(float(total) / 3, abs=1e-5)


def test_scores_do_not_depend_on_padding_side_or_chunking(question, image):
    reference = _scorer().score_question(question, image)
    left = _scorer(processor=StubProcessor(padding_side="left")).score_question(question, image)
    assert left == pytest.approx(reference, abs=1e-6)
    for chunk in (1, 2, 3):
        processor = StubProcessor()
        chunked = _scorer(processor=processor, max_batch_candidates=chunk)
        assert chunked.score_question(question, image) == pytest.approx(reference, abs=1e-6)
        assert [len(call["text"]) for call in processor.calls] == [chunk] * (4 // chunk) + (
            [4 % chunk] if 4 % chunk else []
        )


def test_whole_alphabet_is_scored_in_one_batch_with_one_image_per_candidate(question, image):
    processor = StubProcessor()
    _scorer(processor=processor).score_question(question, image)
    assert len(processor.calls) == 1
    assert len(processor.calls[0]["images"]) == 4 and processor.calls[0]["videos"] is None


def test_chat_template_prompt_has_no_separator(question, image):
    scorer = _scorer(processor=StubProcessor(chat=True))
    prompt, separator = scorer.build_prompt(question, "image")
    assert prompt.startswith("<user> <image> ") and prompt.endswith("<assistant>\n")
    assert separator == ""
    scores = scorer.score_question(question, image)
    for score, answer in zip(scores, question.role.alphabet, strict=True):
        assert score == pytest.approx(_expected(scorer, prompt, answer), abs=1e-5)
    forced = _scorer(processor=StubProcessor(chat=True), answer_separator="space")
    assert forced.build_prompt(question, "image")[1] == " "
    plain = _scorer(answer_separator="none")
    assert plain.build_prompt(question, "image")[1] == ""


def test_video_frames_fall_back_to_images_for_processors_without_videos(question, tmp_path):
    for index in range(2):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / f"{index}.png")
    media = {"type": "video", "path": str(tmp_path)}
    with_videos = StubProcessor()
    _scorer(processor=with_videos).score_question(question, media)
    assert with_videos.calls[0]["videos"] is not None and with_videos.calls[0]["images"] is None
    without_videos = StubProcessor(video_modality=False)
    scores = _scorer(processor=without_videos).score_question(question, media)
    assert len(without_videos.calls[0]["images"][0]) == 2 and len(scores) == 4


def test_instruction_lists_the_whole_alphabet(question):
    text = build_instruction(question)
    assert text == PROMPT_TEMPLATE.format(
        question="What colour is the cube?", alphabet="red, dark red, very dark red, blue"
    )
    assert text.endswith("Answer:")


def test_find_last_subsequence():
    assert find_last_subsequence([1, 2, 3, 1, 2], [1, 2]) == 3
    assert find_last_subsequence([1, 2, 3], [2, 3]) == 1
    assert find_last_subsequence([1, 2, 3], [3, 2]) is None
    assert find_last_subsequence([1], [1, 2]) is None
    assert find_last_subsequence([1, 2], []) is None


def test_answer_positions_use_the_last_occurrence_and_skip_padding():
    # The answer token 7 also occurs inside the prompt (alphabet listing); the last one is scored.
    right_padded = answer_token_positions([5, 7, 6, 7, 8, 0, 0], [1, 1, 1, 1, 1, 0, 0], [7, 8])
    assert right_padded == [3, 4]
    left_padded = answer_token_positions([0, 0, 5, 7, 6, 7, 8], [0, 0, 1, 1, 1, 1, 1], [7, 8])
    assert left_padded == [5, 6]
    with pytest.raises(RuntimeError, match="locate"):
        answer_token_positions([5, 6], [1, 1], [9])
    with pytest.raises(RuntimeError, match="causal"):
        answer_token_positions([7, 8], [1, 1], [7, 8])


def test_mean_answer_logprob_hand_computed():
    logits = torch.zeros((4, 3))
    logits[1] = torch.log(torch.tensor([0.5, 0.25, 0.25]))  # predicts the token at position 2
    logits[2] = torch.log(torch.tensor([0.1, 0.1, 0.8]))  # predicts the token at position 3
    input_ids = torch.tensor([0, 0, 1, 2])
    value = mean_answer_logprob(logits, input_ids, [2, 3])
    assert value == pytest.approx((np.log(0.25) + np.log(0.8)) / 2, abs=1e-6)
    assert mean_answer_logprob(logits, input_ids, [3]) == pytest.approx(np.log(0.8), abs=1e-6)


def test_length_mismatch_between_logits_and_inputs_is_reported(question, image):
    scorer = _scorer(model=BigramModel(drop_last=True))
    with pytest.raises(RuntimeError, match="differ in length"):
        scorer.score_question(question, image)


def test_options_and_constructor_validation():
    assert ModelLoadOptions("m").trust_remote_code is False
    with pytest.raises(ValueError):
        ModelLoadOptions("m", max_batch_candidates=-1)
    with pytest.raises(ValueError):
        ModelLoadOptions("m", answer_separator="tab")
    with pytest.raises(ValueError):
        HuggingFaceVLMScorer(ModelLoadOptions("m"), processor=StubProcessor())
