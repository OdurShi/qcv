"""Shared fixtures: compiled instances, hand-built artifacts and a small synthetic run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qcv.calibration import question_subsets
from qcv.codebook import build_codebook
from qcv.compilers import CompilerRegistry
from qcv.pipeline import (
    calibrate_jsonl,
    compile_jsonl,
    score_jsonl_synthetic,
    synthesize_raw_jsonl,
)
from qcv.signatures import role_keys
from qcv.synthetic import SyntheticConfig
from qcv.types import (
    CalibrationArtifacts,
    CompiledInstance,
    ConstraintSpec,
    CovarianceModel,
    QuestionSpec,
    RoleSpec,
)

EXAMPLE_QUESTIONS = {
    "clevr": [
        "How many red cubes are there?",
        "Are there any red cubes?",
        "What is the spatial relation of cube a relative to cube b?",
        "Is cube a left of cube b?",
    ],
    "gqa": [
        "Are there any red cars?",
        "How many red cars are there?",
        "What is the spatial relation of car a relative to car b?",
    ],
    "chartqa": [
        "How many more bicycles than motorcycles are there?",
        "What is the signed difference between series a and series b",
        "What is the sum of series a and series b",
        "Is series a greater than series b",
    ],
    "agqa": [
        "How many times did the person open the door?",
        "Did the person open the door happen",
        "Did event a happen before event b",
        "What is the temporal relation of event a to event b",
    ],
    "synthetic": [
        "How many more red cubes than blue spheres are there?",
        "What is the spatial relation of the red cube relative to the blue sphere?",
    ],
}
ALL_EXAMPLES = [(task, text) for task, texts in EXAMPLE_QUESTIONS.items() for text in texts]


@pytest.fixture(scope="session")
def registry() -> CompilerRegistry:
    return CompilerRegistry.from_directory()


@pytest.fixture(scope="session")
def compile_question(registry):
    def compile_one(task: str, question: str, instance_id: str = "x") -> CompiledInstance:
        result = registry.compiler_for(task).compile(instance_id, question)
        assert result.compiled is not None, result.reason
        return result.compiled

    return compile_one


def integer_role(name: str, start: int, stop: int) -> RoleSpec:
    alphabet = tuple(str(value) for value in range(start, stop + 1))
    return RoleSpec(name=name, alphabet=alphabet, answer_type="numeric")


@pytest.fixture(scope="session")
def figure3_instance() -> CompiledInstance:
    """Counting-difference example of the paper's pipeline figure: d = b - m, t = b + m."""
    target = QuestionSpec(
        "q0",
        "How many more bicycles than motorcycles?",
        "d",
        integer_role("difference", -2, 2),
        0,
        is_target=True,
    )
    auxiliaries = (
        QuestionSpec("q0001", "# bicycles?", "b", integer_role("bicycles", 0, 2), 1),
        QuestionSpec("q0002", "# motorcycles?", "m", integer_role("motorcycles", 0, 2), 2),
        QuestionSpec("q0003", "# two-wheelers?", "t", integer_role("total", 0, 4), 3),
    )
    constraints = (
        ConstraintSpec("difference_eq", ("b", "m", "d")),
        ConstraintSpec("sum_eq", ("b", "m", "t")),
    )
    return CompiledInstance("fig3", "toy", target, auxiliaries, constraints, "count_difference")


def make_covariance_model(codebook, covariance: np.ndarray | None = None) -> CovarianceModel:
    dimension = codebook.dimension
    covariance = np.eye(dimension) if covariance is None else covariance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return CovarianceModel(
        signature=codebook.signature,
        role_keys=role_keys(codebook.questions),
        dimension=dimension,
        sample_count=100,
        alpha_lw=0.0,
        alpha=0.01,
        mu=float(np.trace(covariance) / dimension),
        covariance=covariance,
        whitening=whitening,
    )


def identity_artifacts(compiled: CompiledInstance, max_auxiliaries: int | None = None):
    """Artifacts with unit temperatures and identity covariance for every subset signature."""
    limit = len(compiled.auxiliaries) if max_auxiliaries is None else max_auxiliaries
    covariances = {}
    for questions in question_subsets(compiled, limit):
        codebook = build_codebook(compiled, questions)
        covariances[codebook.signature] = make_covariance_model(codebook)
    roles = {question.role.key for question in compiled.all_questions}
    return CalibrationArtifacts(
        temperatures=dict.fromkeys(roles, 1.0),
        role_accuracy=dict.fromkeys(roles, 0.5),
        covariances=covariances,
    )


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory) -> dict[str, Path]:
    """Small synthetic calibration/test run produced by the real pipeline stages."""
    root = tmp_path_factory.mktemp("synthetic_run")
    config = SyntheticConfig(seed=0)
    synthesize_raw_jsonl(root, calibration_size=500, test_size=600, config=config)
    for split in ("calibration", "test"):
        compile_jsonl(root / f"{split}_raw.jsonl", root / f"{split}_compiled.jsonl")
        score_jsonl_synthetic(
            root / f"{split}_compiled.jsonl", root / f"{split}_scores.jsonl", config
        )
    calibrate_jsonl(root / "calibration_scores.jsonl", root / "artifacts", max_budget=5)
    return {
        "root": root,
        "calibration_scores": root / "calibration_scores.jsonl",
        "scores": root / "test_scores.jsonl",
        "artifacts": root / "artifacts",
    }


@pytest.fixture(scope="session")
def run_variant(synthetic_run):
    """Run ``qcv infer`` + ``qcv evaluate`` through the CLI once per argument tuple."""
    from qcv.cli import main

    cache: dict[tuple[str, ...], tuple[dict, Path]] = {}

    def run(*arguments: str) -> tuple[dict, Path]:
        if arguments not in cache:
            stem = synthetic_run["root"] / f"variant_{len(cache)}"
            predictions, metrics = stem.with_suffix(".jsonl"), stem.with_suffix(".json")
            common = ["--artifacts", str(synthetic_run["artifacts"])]
            scores = str(synthetic_run["scores"])
            main(["infer", "--input", scores, "--output", str(predictions), *common, *arguments])
            evaluate = ["evaluate", "--scores", scores, "--predictions", str(predictions)]
            main([*evaluate, "--output", str(metrics), *common])
            cache[arguments] = (json.loads(metrics.read_text(encoding="utf-8")), predictions)
        return cache[arguments]

    return run


def random_artifacts(compiled: CompiledInstance, seed: int = 0) -> CalibrationArtifacts:
    """Artifacts with a random well-conditioned covariance for every subset signature."""
    generator = np.random.default_rng(seed)
    covariances = {}
    for questions in question_subsets(compiled, len(compiled.auxiliaries)):
        codebook = build_codebook(compiled, questions)
        if codebook.signature in covariances:
            continue
        mixing = generator.normal(size=(codebook.dimension, codebook.dimension))
        covariance = mixing @ mixing.T / codebook.dimension + 0.5 * np.eye(codebook.dimension)
        covariances[codebook.signature] = make_covariance_model(codebook, covariance)
    roles = {question.role.key for question in compiled.all_questions}
    return CalibrationArtifacts(
        temperatures=dict.fromkeys(roles, 1.0),
        role_accuracy=dict.fromkeys(roles, 0.5),
        covariances=covariances,
    )


def make_record(
    compiled: CompiledInstance,
    believed: dict[str, str],
    strength: float | dict[str, float] = 4.0,
    gold: dict[str, str] | None = None,
    target_answer: str | None = None,
):
    """Score record whose log-scores put ``strength`` on the believed answer of each variable."""
    from qcv.types import QuestionScores, ScoreRecord

    scores = {}
    for question in compiled.all_questions:
        logits = np.zeros(len(question.role.alphabet))
        boost = strength[question.variable] if isinstance(strength, dict) else strength
        logits[question.role.answer_index(believed[question.variable])] = boost
        log_scores = logits - np.log(np.sum(np.exp(logits)))
        scores[question.question_id] = QuestionScores(question.question_id, tuple(log_scores))
    return ScoreRecord(
        instance_id=compiled.instance_id,
        task=compiled.task,
        compiled=compiled,
        scores=scores,
        gold=gold or {},
        target_answer=target_answer,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
