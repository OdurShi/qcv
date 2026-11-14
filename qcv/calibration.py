"""Offline calibration: role temperatures, role reliabilities and residual covariances.

Everything here runs once on a held-out calibration split whose records carry
the true answers of the typed variables (``gold``). The fitted artifacts are
frozen and are the only calibration information available at test time.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .codebook import Codebook, build_codebook
from .covariance import fit_residual_covariance
from .signatures import role_keys
from .temperature import calibrated_probabilities, fit_temperature
from .types import CalibrationArtifacts, CompiledInstance, QuestionSpec, ScoreRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationSummary:
    records_seen: int
    role_models: int
    covariance_models: int
    unsupported_signatures: int
    skipped_incomplete_assignments: int


def probabilities_for_record(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    questions: Sequence[QuestionSpec] | None = None,
) -> dict[str, np.ndarray]:
    """Calibrated distributions ``p_k`` (Eq. 2) of ``questions`` (default: the whole pool)."""
    return {
        question.question_id: calibrated_probabilities(
            record.scores[question.question_id].mean_logprobs,
            artifacts.temperature_for(question.role),
        )
        for question in (record.compiled.all_questions if questions is None else questions)
    }


def question_subsets(
    compiled: CompiledInstance, max_auxiliaries: int, min_auxiliaries: int = 0
) -> Iterator[tuple[QuestionSpec, ...]]:
    """Every question set ``{q0} ∪ Q`` with ``min <= |Q| <= max`` in compiler order."""
    upper = min(max_auxiliaries, len(compiled.auxiliaries))
    for size in range(min_auxiliaries, upper + 1):
        for subset in itertools.combinations(compiled.auxiliaries, size):
            yield (compiled.target, *subset)


def _fit_role_models(
    records: Sequence[ScoreRecord],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float | int | bool]]]:
    """Temperature (Eq. 2) and arg-max accuracy of every role with gold observations."""
    role_scores: dict[str, list[tuple[tuple[float, ...], int]]] = defaultdict(list)
    for record in records:
        for question in record.compiled.all_questions:
            if question.variable not in record.gold:
                continue
            try:
                gold_index = question.role.answer_index(str(record.gold[question.variable]))
            except ValueError:
                continue
            role_scores[question.role.key].append(
                (record.scores[question.question_id].mean_logprobs, gold_index)
            )

    temperatures: dict[str, float] = {}
    role_accuracy: dict[str, float] = {}
    details: dict[str, dict[str, float | int | bool]] = {}
    for role_key, observations in sorted(role_scores.items()):
        scores = np.asarray([item[0] for item in observations], dtype=np.float64)
        gold = np.asarray([item[1] for item in observations], dtype=np.int64)
        fit = fit_temperature(scores, gold)
        temperatures[role_key] = fit.temperature
        role_accuracy[role_key] = float(np.mean(np.argmax(scores, axis=1) == gold))
        details[role_key] = {
            "temperature": fit.temperature,
            "nll": fit.nll,
            "converged": fit.converged,
            "iterations": fit.iterations,
            "observations": len(observations),
        }
    return temperatures, role_accuracy, details


def fit_calibration(
    records: Sequence[ScoreRecord],
    max_budget: int,
) -> tuple[CalibrationArtifacts, CalibrationSummary]:
    """Fit all calibration artifacts for budgets up to ``max_budget``.

    For every record and every question set with at most ``max_budget - 1``
    auxiliaries, the residual ``r = z - c`` between the stacked calibrated
    observation and the true feasible codeword is accumulated under the set's
    ordered role signature; one shrunk covariance is then fitted per signature
    (Eq. 3). Records whose gold answers are incomplete for a set, or whose gold
    tuple is not a feasible codeword, do not contribute to that signature.
    """
    if max_budget < 1:
        raise ValueError("max_budget must be at least one")
    if not records:
        raise ValueError("No calibration records were provided")

    temperatures, role_accuracy, temperature_details = _fit_role_models(records)

    residuals_by_signature: dict[str, list[np.ndarray]] = defaultdict(list)
    role_keys_by_signature: dict[str, tuple[str, ...]] = {}
    plans: dict[str, list[tuple[Codebook, tuple[int, ...], np.ndarray]]] = {}
    skipped_incomplete = 0
    for record in records:
        compiled = record.compiled
        pool = compiled.all_questions
        plan = plans.get(compiled.structure_key)
        if plan is None:
            # Per structure: codebook, pool positions and residual columns of every question set.
            offsets = np.concatenate([[0], np.cumsum([len(q.role.alphabet) for q in pool])])
            position = {question.question_id: index for index, question in enumerate(pool)}
            plan = []
            for questions in question_subsets(compiled, max_budget - 1):
                codebook = build_codebook(compiled, questions)
                members = tuple(position[question.question_id] for question in questions)
                columns = np.concatenate(
                    [np.arange(offsets[index], offsets[index + 1]) for index in members]
                )
                plan.append((codebook, members, columns))
                role_keys_by_signature.setdefault(codebook.signature, role_keys(questions))
            plans[compiled.structure_key] = plan

        # Residual r = z - c of the whole pool; question sets are column slices of it.
        gold_indices: list[int | None] = []
        residual_blocks: list[np.ndarray] = []
        for question in pool:
            gold_index = None
            block = np.zeros(len(question.role.alphabet), dtype=np.float64)
            if question.role.key in temperatures and question.variable in record.gold:
                try:
                    gold_index = question.role.answer_index(str(record.gold[question.variable]))
                except ValueError:
                    gold_index = None
            if gold_index is not None:
                block = calibrated_probabilities(
                    record.scores[question.question_id].mean_logprobs,
                    temperatures[question.role.key],
                )
                block[gold_index] -= 1.0
            gold_indices.append(gold_index)
            residual_blocks.append(block)
        pool_residual = np.concatenate(residual_blocks)

        for codebook, members, columns in plan:
            if codebook.size == 0:
                continue
            indices = tuple(gold_indices[index] for index in members)
            if None in indices or codebook.row_of_indices(indices) is None:
                skipped_incomplete += 1
                continue
            residuals_by_signature[codebook.signature].append(pool_residual[columns])

    covariances = {}
    covariance_failures: dict[str, str] = {}
    for signature, signature_role_keys in sorted(role_keys_by_signature.items()):
        residual_rows = residuals_by_signature.get(signature, [])
        if not residual_rows:
            covariance_failures[signature] = "no complete calibration residuals"
            continue
        try:
            covariances[signature] = fit_residual_covariance(
                signature=signature,
                role_keys=signature_role_keys,
                residual_rows=np.stack(residual_rows, axis=0),
            )
        except ValueError as exc:
            covariance_failures[signature] = str(exc)
    if covariance_failures:
        logger.warning(
            "%d role signatures are outside QCV coverage (see metadata.json)",
            len(covariance_failures),
        )

    metadata = {
        "method": "QCV",
        "max_budget": max_budget,
        "records_seen": len(records),
        "temperature_details": temperature_details,
        "covariance_failures": covariance_failures,
        "covariance_sample_counts": {
            signature: model.sample_count for signature, model in covariances.items()
        },
    }
    artifacts = CalibrationArtifacts(
        temperatures=temperatures,
        role_accuracy=role_accuracy,
        covariances=covariances,
        metadata=metadata,
    )
    summary = CalibrationSummary(
        records_seen=len(records),
        role_models=len(temperatures),
        covariance_models=len(covariances),
        unsupported_signatures=len(covariance_failures),
        skipped_incomplete_assignments=skipped_incomplete,
    )
    return artifacts, summary
