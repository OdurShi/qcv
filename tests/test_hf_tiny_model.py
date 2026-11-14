"""The scorer on real ``transformers`` classes: a tiny randomly initialised LLaVA model.

Nothing is downloaded. The processor (character-level tokenizer, CLIP image
processor, chat template) and the model (two-layer CLIP vision tower and
two-layer LLaMA decoder) are built in code. The scores are meaningless, but the
whole code path is the one a pretrained VLM takes: chat-template rendering,
image-token expansion, padded batching, a forward pass with pixel values, and
loading through the ``qcv score`` command.
"""

from __future__ import annotations

import string

import numpy as np
import pytest
from PIL import Image

from qcv.cli import main
from qcv.models.hf import HuggingFaceVLMScorer, ModelLoadOptions
from qcv.pipeline import load_predictions, load_score_records
from qcv.synthetic import SyntheticConfig, generate_raw_records
from qcv.types import QuestionSpec, RoleSpec
from qcv.utils import write_jsonl

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'].upper() }}: "
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{ item['text'] }}{% else %}<image> {% endif %}"
    "{% endfor %} {% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
)


def build_tiny_llava():
    specials = ["<pad>", "<unk>", "<s>", "</s>", "<image>"]
    characters = list(string.ascii_letters + string.digits + string.punctuation + " \n")
    vocabulary = {token: index for index, token in enumerate(specials + characters)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex(r"[\s\S]"), behavior="isolated"
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["<image>"],
    )
    processor = transformers.LlavaProcessor(
        image_processor=transformers.CLIPImageProcessor(
            size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28}
        ),
        tokenizer=tokenizer,
        patch_size=14,
        vision_feature_select_strategy="default",
        chat_template=CHAT_TEMPLATE,
        image_token="<image>",
        num_additional_image_tokens=1,
    )
    config = transformers.LlavaConfig(
        vision_config=transformers.CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        ),
        text_config=transformers.LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        ),
        image_token_index=vocabulary["<image>"],
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
    )
    torch.manual_seed(0)
    return processor, transformers.LlavaForConditionalGeneration(config).eval()


@pytest.fixture(scope="module")
def tiny_llava():
    return build_tiny_llava()


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "scene.png"
    pixels = np.random.default_rng(0).integers(0, 255, (40, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return {"type": "image", "path": str(path)}


@pytest.fixture(scope="module")
def question():
    role = RoleSpec("count", ("0", "1", "2", "10", "11", "none of these"), "categorical")
    return QuestionSpec("q0", "How many red cubes are there?", "count_a", role, 0, is_target=True)


def _scorer(tiny_llava, **options):
    processor, model = tiny_llava
    return HuggingFaceVLMScorer(
        ModelLoadOptions("tiny-llava", device="cpu", dtype="float32", **options),
        processor=processor,
        model=model,
    )


def test_scores_match_an_unbatched_forward_pass(tiny_llava, question, image):
    processor, model = tiny_llava
    scorer = _scorer(tiny_llava)
    prompt, separator = scorer.build_prompt(question, "image")
    assert "<image>" in prompt and prompt.endswith("ASSISTANT: ") and separator == ""
    assert "[0, 1, 2, 10, 11, none of these]" in prompt
    scores = scorer.score_question(question, image)
    assert len(scores) == 6 and np.all(np.isfinite(scores))

    picture = Image.open(image["path"]).convert("RGB")
    for score, answer in zip(scores, question.role.alphabet, strict=True):
        encoded = processor(text=[prompt + answer], images=[picture], return_tensors="pt")
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
        input_ids = encoded["input_ids"][0]
        # The image placeholder is expanded to one token per patch: lengths must agree.
        assert logits.shape[0] == input_ids.shape[0]
        length = len(processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
        assert length == len(answer)  # character-level tokens
        positions = torch.arange(input_ids.shape[0] - length, input_ids.shape[0])
        log_probabilities = torch.log_softmax(logits[positions - 1], dim=-1)
        expected = log_probabilities.gather(1, input_ids[positions][:, None]).mean()
        assert score == pytest.approx(float(expected), abs=1e-4)


def test_padding_side_and_chunking_do_not_change_real_model_scores(tiny_llava, question, image):
    processor, _ = tiny_llava
    reference = _scorer(tiny_llava).score_question(question, image)
    assert _scorer(tiny_llava, max_batch_candidates=2).score_question(
        question, image
    ) == pytest.approx(reference, abs=1e-4)
    processor.tokenizer.padding_side = "left"
    try:
        left = _scorer(tiny_llava).score_question(question, image)
    finally:
        processor.tokenizer.padding_side = "right"
    assert left == pytest.approx(reference, abs=1e-4)


def test_score_command_runs_end_to_end_with_a_locally_saved_model(tiny_llava, tmp_path, capsys):
    processor, model = tiny_llava
    model_directory = tmp_path / "tiny-llava"
    model.save_pretrained(model_directory)
    processor.save_pretrained(model_directory)

    generator = np.random.default_rng(1)
    rows = generate_raw_records(8, "test", SyntheticConfig(seed=2))
    for index, row in enumerate(rows):
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(generator.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
        row["media"] = {"type": "image", "path": str(path)}
    write_jsonl(tmp_path / "raw.jsonl", rows)

    def run(*arguments):
        main([str(argument) for argument in arguments])

    run("compile", "--input", tmp_path / "raw.jsonl", "--output", tmp_path / "compiled.jsonl")
    run("score", "--backend", "hf", "--model", model_directory, "--device", "cpu",
        "--dtype", "float32", "--input", tmp_path / "compiled.jsonl",
        "--output", tmp_path / "scores.jsonl")  # fmt: skip
    run("calibrate", "--input", tmp_path / "scores.jsonl",
        "--output-dir", tmp_path / "artifacts", "--max-budget", "3")  # fmt: skip
    run("infer", "--input", tmp_path / "scores.jsonl", "--artifacts", tmp_path / "artifacts",
        "--output", tmp_path / "predictions.jsonl", "--budget", "3")  # fmt: skip
    capsys.readouterr()

    records = load_score_records(tmp_path / "scores.jsonl")
    assert len(records) == 8
    for record in records:
        assert record.metadata["model"] == str(model_directory)
        for item in record.compiled.all_questions:
            values = np.asarray(record.scores[item.question_id].mean_logprobs)
            assert values.shape == (len(item.role.alphabet),)
            assert np.all(np.isfinite(values)) and np.all(values < 0.0)
    predictions = load_predictions(tmp_path / "predictions.jsonl")
    assert len(predictions) == 8
    assert all(p.final_answer in r.compiled.target.role.alphabet
               for p, r in zip(predictions, records, strict=True))  # fmt: skip
