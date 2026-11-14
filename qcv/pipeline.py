"""File-level pipeline stages behind the ``qcv`` command line interface.

compile -> score -> calibrate -> infer -> evaluate -> compare, each reading and
writing JSON(L) files so that every stage can be cached and audited.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baselines import baseline_prediction
from .calibration import fit_calibration
from .compilers import CompilerRegistry
from .data import default_data_dir, validate_raw_record
from .data.download import sha256_of
from .inference import InferenceEngine, InferenceOptions
from .metrics import evaluate_predictions
from .synthetic import SyntheticConfig, SyntheticScorer, generate_raw_records
from .types import (
    CalibrationArtifacts,
    CompiledInstance,
    Prediction,
    QuestionScores,
    QuestionSpec,
    ScoreRecord,
)
from .utils import json_dump, json_load, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

QuestionScorer = Callable[[QuestionSpec, CompiledInstance, dict[str, Any]], tuple[float, ...]]
"""Maps (question, compiled instance, raw row) to log-scores of the whole answer alphabet."""


def compile_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    grammar_directory: str | Path | None = None,
    keep_uncovered: bool = True,
) -> dict[str, int]:
    """Compile raw records; rows that do not compile are kept (flagged) unless dropped."""
    registry = CompilerRegistry.from_directory(grammar_directory)
    output: list[dict[str, Any]] = []
    compiled_count = 0
    rejected_count = 0
    for row in read_jsonl(input_path):
        enriched = validate_raw_record(row)
        result = registry.compiler_for(enriched["task"]).compile(
            enriched["instance_id"], enriched["question"]
        )
        enriched["compile_status"] = result.reason
        enriched["matching_rules"] = list(result.matching_rules)
        if result.compiled is not None:
            enriched["compiled"] = result.compiled.to_dict()
            output.append(enriched)
            compiled_count += 1
        else:
            rejected_count += 1
            if keep_uncovered:
                output.append(enriched)
    write_jsonl(output_path, output)
    return {"compiled": compiled_count, "rejected": rejected_count, "written": len(output)}


def _score_rows(
    input_path: str | Path,
    output_path: str | Path,
    scorer: QuestionScorer,
    extra_metadata: dict[str, Any],
) -> dict[str, int]:
    """Score the target and every pool auxiliary of each compiled row.

    The whole pool is scored so that one score file serves every budget and
    selection variant; inference later reads exactly ``B`` of these blocks.
    Rows without a compiled instance have no typed answer alphabet and are skipped.
    """
    output: list[dict[str, Any]] = []
    skipped = 0
    for row in read_jsonl(input_path):
        if "compiled" not in row:
            skipped += 1
            continue
        compiled = CompiledInstance.from_dict(row["compiled"])
        scores: dict[str, QuestionScores] = {}
        latencies: dict[str, float] = {}
        for question in compiled.all_questions:
            start = time.perf_counter()
            values = scorer(question, compiled, row)
            latencies[question.question_id] = time.perf_counter() - start
            scores[question.question_id] = QuestionScores(question.question_id, values)
        record = ScoreRecord(
            instance_id=compiled.instance_id,
            task=compiled.task,
            compiled=compiled,
            scores=scores,
            gold={str(k): str(v) for k, v in row.get("gold", {}).items()},
            target_answer=(None if row.get("answer") is None else str(row["answer"])),
            media=dict(row.get("media", {})),
            metadata={
                **dict(row.get("metadata", {})),
                **extra_metadata,
                "question_latency_seconds": latencies,
            },
        )
        output.append(record.to_dict())
    write_jsonl(output_path, output)
    if skipped:
        logger.warning("%d rows without a compiled instance were not scored", skipped)
    return {"scored": len(output), "skipped": skipped}


def score_jsonl_hf(
    input_path: str | Path,
    output_path: str | Path,
    model_options: Any,
) -> dict[str, int]:
    """Score compiled rows with a frozen Hugging Face VLM (Eq. 2)."""
    from .models import HuggingFaceVLMScorer

    vlm = HuggingFaceVLMScorer(model_options)

    def scorer(
        question: QuestionSpec, compiled: CompiledInstance, row: dict[str, Any]
    ) -> tuple[float, ...]:
        media = dict(row.get("media", {}))
        if not media:
            raise ValueError(f"Record {row.get('instance_id')} lacks media information")
        return vlm.score_question(question, media)

    metadata = {"model": model_options.model_name_or_path, "dtype": model_options.dtype}
    return _score_rows(input_path, output_path, scorer, metadata)


def score_jsonl_synthetic(
    input_path: str | Path,
    output_path: str | Path,
    config: SyntheticConfig,
) -> dict[str, int]:
    """Score compiled rows with the synthetic simulator instead of a VLM.

    The simulator reads the latent scene state from each row's ``gold`` field,
    which stands in for the image a real VLM would look at.
    """
    simulator = SyntheticScorer(config)

    def scorer(
        question: QuestionSpec, compiled: CompiledInstance, row: dict[str, Any]
    ) -> tuple[float, ...]:
        gold = {str(key): str(value) for key, value in row.get("gold", {}).items()}
        if any(item.variable not in gold for item in (question, compiled.target)):
            raise ValueError(f"Record {compiled.instance_id} has no complete scene state")
        return simulator.score_question(question, compiled.target, gold, compiled.instance_id)

    metadata = {"model": "synthetic-simulator", "simulator_seed": config.seed}
    return _score_rows(input_path, output_path, scorer, metadata)


SYNTHETIC_SPLITS = ("calibration", "test")


def synthesize_raw_jsonl(
    output_directory: str | Path,
    calibration_size: int,
    test_size: int,
    config: SyntheticConfig,
) -> dict[str, Any]:
    """Write ``calibration_raw.jsonl``, ``test_raw.jsonl`` and ``MANIFEST.json``.

    The raw records (questions and scene states) depend only on the seed, the
    split sizes and ``relation_fraction``; the simulated scores are produced
    later by ``qcv score --backend synthetic``.
    """
    directory = Path(output_directory)
    sizes = {"calibration": calibration_size, "test": test_size}
    files: dict[str, Any] = {}
    for split in SYNTHETIC_SPLITS:
        path = directory / f"{split}_raw.jsonl"
        write_jsonl(path, generate_raw_records(sizes[split], split, config))
        files[path.name] = {"records": sizes[split], "sha256": sha256_of(path)}
    manifest = {
        "generator": "qcv synth",
        "seed": config.seed,
        "relation_fraction": config.relation_fraction,
        "calibration_size": calibration_size,
        "test_size": test_size,
        "files": files,
    }
    json_dump(directory / "MANIFEST.json", manifest)
    return {
        "calibration": str(directory / "calibration_raw.jsonl"),
        "calibration_records": calibration_size,
        "test": str(directory / "test_raw.jsonl"),
        "test_records": test_size,
    }


def provide_synthetic_raw_jsonl(
    output_directory: str | Path,
    calibration_size: int,
    test_size: int,
    config: SyntheticConfig,
    data_dir: str | Path | None = None,
) -> str:
    """Place the synthetic raw records in ``output_directory``.

    The records shipped in ``<data_dir>/synthetic`` are copied when their
    manifest matches the requested seed, sizes and ``relation_fraction``;
    otherwise the records are generated. Returns ``"data"`` or ``"generated"``.
    """
    directory = Path(output_directory)
    shipped = default_data_dir(data_dir) / "synthetic"
    manifest_path = shipped / "MANIFEST.json"
    if manifest_path.is_file() and shipped.resolve() != directory.resolve():
        manifest = json_load(manifest_path)
        wanted = {
            "seed": config.seed,
            "relation_fraction": config.relation_fraction,
            "calibration_size": calibration_size,
            "test_size": test_size,
        }
        if all(manifest.get(key) == value for key, value in wanted.items()):
            directory.mkdir(parents=True, exist_ok=True)
            for split in SYNTHETIC_SPLITS:
                shutil.copyfile(shipped / f"{split}_raw.jsonl", directory / f"{split}_raw.jsonl")
            return "data"
    synthesize_raw_jsonl(directory, calibration_size, test_size, config)
    return "generated"


def load_score_records(path: str | Path) -> list[ScoreRecord]:
    return [ScoreRecord.from_dict(row) for row in read_jsonl(path)]


def load_predictions(path: str | Path) -> list[Prediction]:
    return [Prediction.from_dict(row) for row in read_jsonl(path)]


def calibrate_jsonl(
    input_path: str | Path,
    output_directory: str | Path,
    max_budget: int,
) -> dict[str, Any]:
    """Fit temperatures and residual covariances on a calibration score file."""
    records = load_score_records(input_path)
    artifacts, summary = fit_calibration(records, max_budget=max_budget)
    artifacts.save(output_directory)
    result = {
        "records_seen": summary.records_seen,
        "role_models": summary.role_models,
        "covariance_models": summary.covariance_models,
        "unsupported_signatures": summary.unsupported_signatures,
        "skipped_incomplete_assignments": summary.skipped_incomplete_assignments,
    }
    json_dump(Path(output_directory) / "calibration_summary.json", result)
    return result


def _coverage_counts(predictions: list[Prediction]) -> dict[str, int]:
    covered = sum(prediction.covered for prediction in predictions)
    return {
        "predictions": len(predictions),
        "covered": covered,
        "uncovered": len(predictions) - covered,
    }


def infer_jsonl(
    input_path: str | Path,
    artifact_directory: str | Path,
    output_path: str | Path,
    options: InferenceOptions,
) -> dict[str, int]:
    """Run QCV (or a QCV control) on a test score file."""
    engine = InferenceEngine(CalibrationArtifacts.load(artifact_directory), options)
    predictions = [engine.infer(record) for record in load_score_records(input_path)]
    write_jsonl(output_path, [prediction.to_dict() for prediction in predictions])
    return _coverage_counts(predictions)


def baseline_jsonl(
    input_path: str | Path,
    artifact_directory: str | Path,
    output_path: str | Path,
    method: str,
    budget: int = 5,
    seed: int = 0,
) -> dict[str, int]:
    """Run a score-file baseline (target-only or repeated-target) on a test score file."""
    artifacts = CalibrationArtifacts.load(artifact_directory)
    predictions = [
        baseline_prediction(record, artifacts, method, budget=budget, seed=seed)
        for record in load_score_records(input_path)
    ]
    write_jsonl(output_path, [prediction.to_dict() for prediction in predictions])
    return _coverage_counts(predictions)


def evaluate_jsonl(
    score_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    artifact_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a prediction file against the annotations stored in the score file."""
    records = load_score_records(score_path)
    predictions = load_predictions(prediction_path)
    artifacts = (
        None if artifact_directory is None else CalibrationArtifacts.load(artifact_directory)
    )
    result = evaluate_predictions(records, predictions, artifacts=artifacts).to_dict()
    json_dump(output_path, result)
    return result
