"""Evaluation metrics: accuracy, coverage, correction behaviour, failure geometry, tests.

Definitions (see ``docs/implementation_notes.md`` for the ones the paper's main
text leaves to its appendix):

* overall accuracy       final answers on all examples, fallback predictions included
* coverage               fraction of examples decoded by QCV rather than the fallback
* covered accuracy       accuracy on the covered subset only
* correction rate        P(final correct | target-only answer wrong)
* corruption rate        P(final wrong   | target-only answer correct)
* rho_bar_B              mean pairwise Pearson correlation between the error
                         indicators of the B measurement slots, across examples
* B_eff                  B / (1 + (B - 1) rho_bar_B)
* S_B                    mean fraction of the other B - 1 measurements that are
                         correct on examples whose target measurement is wrong
* error AUROC            how well a confidence score ranks wrong final answers
* kappa = 2 r / d_tar    normalised margin of the deterministic condition (Eq. 8)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
from scipy.stats import binomtest, rankdata

from .calibration import probabilities_for_record
from .codebook import build_codebook
from .geometry import (
    normalized_margin,
    stack_observation,
    whitened_residual_radius,
    whitening_for_mode,
)
from .selection import set_separation
from .types import CalibrationArtifacts, Prediction, ScoreRecord
from .utils import normalize_text

KAPPA_BIN_EDGES = (0.8, 1.2)
"""Bin edges of the normalised margin used in the paper's separation audit."""


def answers_equal(task: str, prediction: str, gold: str) -> bool:
    """Normalised exact match; ChartQA numeric answers use the standard 5% relaxed match."""
    pred = normalize_text(str(prediction))
    truth = normalize_text(str(gold))
    if task.lower() == "chartqa":
        try:
            pred_number = Decimal(pred.replace(",", "").replace("%", ""))
            gold_number = Decimal(truth.replace(",", "").replace("%", ""))
        except InvalidOperation:
            return pred == truth
        if gold_number == 0:
            return pred_number == gold_number
        return abs(pred_number - gold_number) <= abs(gold_number) * Decimal("0.05")
    return pred == truth


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve via the rank-sum statistic (ties get average ranks)."""
    labels_array = np.asarray(labels, dtype=bool)
    positives = int(np.sum(labels_array))
    negatives = int(labels_array.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    rank_sum = float(np.sum(ranks[labels_array]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def mean_pairwise_error_correlation(errors: np.ndarray) -> float | None:
    """``rho_bar_B``: mean Pearson correlation over pairs of measurement slots.

    ``errors`` is an ``(examples, B)`` 0/1 matrix. Pairs in which either slot
    has zero variance (always right or always wrong) are undefined and skipped.
    """
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] < 2:
        return None
    variances = np.var(errors, axis=0)
    correlations: list[float] = []
    for left in range(errors.shape[1]):
        for right in range(left + 1, errors.shape[1]):
            if variances[left] == 0.0 or variances[right] == 0.0:
                continue
            correlations.append(float(np.corrcoef(errors[:, left], errors[:, right])[0, 1]))
    return float(np.mean(correlations)) if correlations else None


def effective_measurements(budget: int, rho: float) -> float:
    """``B_eff = B / (1 + (B - 1) rho_bar_B)``, capped at the nominal budget ``B``."""
    denominator = 1.0 + (budget - 1) * rho
    if denominator <= 0.0:
        return float(budget)
    return min(float(budget), budget / denominator)


def post_failure_survival(errors: np.ndarray) -> float | None:
    """``S_B``: share of the non-target measurements that are correct when the target fails.

    Slot 0 is the target measurement. The statistic averages, over examples with
    a wrong target measurement, the fraction of correct remaining measurements.
    """
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] < 2:
        return None
    failed = errors[:, 0] == 1.0
    if not np.any(failed):
        return None
    return float(np.mean(1.0 - errors[failed, 1:]))


def correction_and_corruption(
    initial_correct: Sequence[bool], final_correct: Sequence[bool]
) -> tuple[float | None, float | None]:
    """Correction rate P(final right | initially wrong), corruption rate P(final wrong | right)."""
    initial = np.asarray(initial_correct, dtype=bool)
    final = np.asarray(final_correct, dtype=bool)
    wrong = int(np.sum(~initial))
    right = int(np.sum(initial))
    correction = float(np.sum(final & ~initial) / wrong) if wrong else None
    corruption = float(np.sum(~final & initial) / right) if right else None
    return correction, corruption


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    overall_accuracy: float
    macro_overall_accuracy: float
    coverage: float
    covered_accuracy: float | None
    uncovered_accuracy: float | None
    target_only_accuracy: float
    correction_rate: float | None
    corruption_rate: float | None
    mean_error_correlation: float | None
    effective_measurements: float | None
    correct_measurement_survival: float | None
    margin_error_auroc: float | None
    gap_error_auroc: float | None
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    separation_audit: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "overall_accuracy": self.overall_accuracy,
            "macro_overall_accuracy": self.macro_overall_accuracy,
            "coverage": self.coverage,
            "covered_accuracy": self.covered_accuracy,
            "uncovered_accuracy": self.uncovered_accuracy,
            "target_only_accuracy": self.target_only_accuracy,
            "correction_rate": self.correction_rate,
            "corruption_rate": self.corruption_rate,
            "mean_error_correlation": self.mean_error_correlation,
            "effective_measurements": self.effective_measurements,
            "correct_measurement_survival": self.correct_measurement_survival,
            "margin_error_auroc": self.margin_error_auroc,
            "gap_error_auroc": self.gap_error_auroc,
            "per_task": self.per_task,
            "separation_audit": self.separation_audit,
        }


def _accuracy_block(final: np.ndarray, initial: np.ndarray, covered: np.ndarray) -> dict[str, Any]:
    correction, corruption = correction_and_corruption(initial, final)
    return {
        "total": int(final.size),
        "overall_accuracy": float(np.mean(final)),
        "coverage": float(np.mean(covered)),
        "covered_accuracy": float(np.mean(final[covered])) if np.any(covered) else None,
        "uncovered_accuracy": float(np.mean(final[~covered])) if np.any(~covered) else None,
        "target_only_accuracy": float(np.mean(initial)),
        "correction_rate": correction,
        "corruption_rate": corruption,
    }


def _measurement_errors(record: ScoreRecord, prediction: Prediction) -> list[int] | None:
    """0/1 error indicator of each measurement, or ``None`` if a gold answer is missing."""
    row: list[int] = []
    for question_id, answer in prediction.measurements:
        question = record.compiled.question_by_id(question_id)
        if question.variable not in record.gold:
            return None
        correct = answers_equal(record.task, answer, record.gold[question.variable])
        row.append(0 if correct else 1)
    return row


def _normalized_margin(
    record: ScoreRecord, prediction: Prediction, artifacts: CalibrationArtifacts
) -> float | None:
    """``kappa = 2 ||W eps||_2 / d_tar`` of a covered prediction (needs gold auxiliary answers)."""
    try:
        questions = record.compiled.ordered_subset(prediction.selected_question_ids)
        codebook = build_codebook(record.compiled, questions)
        true_row = codebook.row_of(record.gold)
        if true_row is None:
            return None
        metric = prediction.metric or "full"
        whitening = whitening_for_mode(artifacts.covariance_for(codebook.signature), metric)
        observation = stack_observation(
            codebook.questions, probabilities_for_record(record, artifacts, questions)
        )
        separation = set_separation(record.compiled, questions, artifacts, metric)
    except (KeyError, ValueError):
        return None
    radius = whitened_residual_radius(observation, codebook.codewords[true_row], whitening)
    return normalized_margin(radius, separation)


def _separation_audit(rows: Sequence[tuple[float, bool, bool]]) -> dict[str, Any] | None:
    """Summarise ``(kappa, target_correct, initially_correct)`` triples (Eq. 8 audit)."""
    if not rows:
        return None
    kappa = np.asarray([row[0] for row in rows], dtype=np.float64)
    correct = np.asarray([row[1] for row in rows], dtype=bool)
    initial = np.asarray([row[2] for row in rows], dtype=bool)
    low, high = KAPPA_BIN_EDGES
    bins = {
        f"kappa<{low}": kappa < low,
        f"{low}<=kappa<{high}": (kappa >= low) & (kappa < high),
        f"kappa>={high}": kappa >= high,
    }
    summary: dict[str, Any] = {
        "examples": int(kappa.size),
        "sufficient_condition_rate": float(np.mean(kappa < 1.0)),
        "sufficient_condition_violations": int(np.sum((kappa < 1.0) & ~correct)),
        "bins": {},
    }
    for name, mask in bins.items():
        correction, _ = correction_and_corruption(initial[mask], correct[mask])
        summary["bins"][name] = {
            "examples": int(np.sum(mask)),
            "final_accuracy": float(np.mean(correct[mask])) if np.any(mask) else None,
            "correction_rate": correction,
        }
    return summary


def evaluate_predictions(
    records: Sequence[ScoreRecord],
    predictions: Sequence[Prediction],
    artifacts: CalibrationArtifacts | None = None,
) -> EvaluationSummary:
    """Compute every reported metric for one prediction file.

    Accuracy-type metrics need only ``target_answer``. The failure-geometry
    metrics additionally need the gold answers of the measured questions, and
    the Eq. (8) audit needs the calibration ``artifacts``; both are analysis-only
    annotations that inference never reads.
    """
    by_id = {prediction.instance_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Prediction IDs are not unique")
    usable_records = [record for record in records if record.target_answer is not None]
    if not usable_records:
        raise ValueError("No target answers are available for evaluation")

    final_correct: list[bool] = []
    target_correct: list[bool] = []
    covered_flags: list[bool] = []
    tasks: list[str] = []
    margin_labels: list[int] = []
    margin_scores: list[float] = []
    gap_labels: list[int] = []
    gap_scores: list[float] = []
    errors_by_width: dict[int, list[list[int]]] = defaultdict(list)
    audit_rows: list[tuple[float, bool, bool]] = []

    for record in usable_records:
        if record.instance_id not in by_id:
            raise KeyError(f"Missing prediction for {record.instance_id}")
        prediction = by_id[record.instance_id]
        gold = str(record.target_answer)
        target_ok = answers_equal(record.task, prediction.target_only_answer, gold)
        final_ok = answers_equal(record.task, prediction.final_answer, gold)
        target_correct.append(target_ok)
        final_correct.append(final_ok)
        covered_flags.append(prediction.covered)
        tasks.append(record.task)
        if not prediction.covered:
            continue

        if prediction.target_margin is not None:
            margin_labels.append(0 if final_ok else 1)
            margin_scores.append(-float(prediction.target_margin))
        if prediction.projection_gap is not None:
            gap_labels.append(0 if final_ok else 1)
            gap_scores.append(float(prediction.projection_gap))

        row = _measurement_errors(record, prediction)
        if row is not None:
            errors_by_width[len(row)].append(row)
        if artifacts is not None and prediction.reason == "qcv":
            kappa = _normalized_margin(record, prediction, artifacts)
            if kappa is not None:
                gold_target = record.gold[record.compiled.target.variable]
                audit_rows.append((kappa, prediction.final_answer == gold_target, target_ok))

    final_array = np.asarray(final_correct, dtype=bool)
    target_array = np.asarray(target_correct, dtype=bool)
    covered_array = np.asarray(covered_flags, dtype=bool)
    overall = _accuracy_block(final_array, target_array, covered_array)
    task_array = np.asarray(tasks)
    per_task = {
        task: _accuracy_block(
            final_array[task_array == task],
            target_array[task_array == task],
            covered_array[task_array == task],
        )
        for task in sorted(set(tasks))
    }

    rho = None
    beff = None
    survival = None
    if len(errors_by_width) == 1:
        matrix = np.asarray(next(iter(errors_by_width.values())), dtype=np.float64)
        rho = mean_pairwise_error_correlation(matrix)
        if rho is not None:
            beff = effective_measurements(matrix.shape[1], rho)
        survival = post_failure_survival(matrix)

    return EvaluationSummary(
        total=len(usable_records),
        overall_accuracy=overall["overall_accuracy"],
        macro_overall_accuracy=float(
            np.mean([block["overall_accuracy"] for block in per_task.values()])
        ),
        coverage=overall["coverage"],
        covered_accuracy=overall["covered_accuracy"],
        uncovered_accuracy=overall["uncovered_accuracy"],
        target_only_accuracy=overall["target_only_accuracy"],
        correction_rate=overall["correction_rate"],
        corruption_rate=overall["corruption_rate"],
        mean_error_correlation=rho,
        effective_measurements=beff,
        correct_measurement_survival=survival,
        margin_error_auroc=auroc(margin_labels, margin_scores),
        gap_error_auroc=auroc(gap_labels, gap_scores),
        per_task=per_task,
        separation_audit=_separation_audit(audit_rows),
    )


def paired_bootstrap_accuracy_difference(
    first_correct: Sequence[bool],
    second_correct: Sequence[bool],
    replicates: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Accuracy difference (first - second) with a paired percentile-bootstrap 95% interval."""
    first = np.asarray(first_correct, dtype=np.float64)
    second = np.asarray(second_correct, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Paired correctness arrays must be one-dimensional and equal length")
    if first.size == 0:
        raise ValueError("Cannot bootstrap an empty comparison")
    paired = first - second
    generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = generator.integers(0, paired.size, size=paired.size)
        differences[index] = np.mean(paired[sample])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(np.mean(paired)), float(lower), float(upper)


def mcnemar_exact(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Paired correctness arrays differ in shape")
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    test = binomtest(min(first_only, second_only), discordant, 0.5, alternative="two-sided")
    return float(test.pvalue)


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for a family of comparisons."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
