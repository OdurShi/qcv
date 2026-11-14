"""Test-time inference protocol: coverage, selection, decoding and the target-only fallback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .calibration import probabilities_for_record, question_subsets
from .codebook import build_codebook
from .decoding import feasible_projection_decode
from .geometry import stack_observation, whitening_for_mode
from .selection import (
    SELECTION_METHODS,
    SelectionResult,
    exact_select,
    fixed_select,
    greedy_select,
    random_select,
    reliability_select,
    set_separation,
)
from .types import (
    METRIC_MODES,
    CalibrationArtifacts,
    CompiledInstance,
    Prediction,
    QuestionSpec,
    ScoreRecord,
)

OPTIMISING_SELECTIONS = frozenset({"qcv", "exact"})


@dataclass(frozen=True)
class InferenceOptions:
    """Inference settings; the defaults are the paper's default QCV configuration."""

    budget: int = 5
    selection: str = "qcv"
    metric: str = "full"
    strict_coverage: bool = True
    random_seed: int = 0
    exact_subset_limit: int = 10_000

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be at least one")
        if self.selection not in SELECTION_METHODS:
            raise ValueError(f"Unsupported selection method {self.selection!r}")
        if self.metric not in METRIC_MODES:
            raise ValueError(f"Unsupported metric mode {self.metric!r}")


def _signature_reason(
    compiled: CompiledInstance,
    questions: Sequence[QuestionSpec],
    artifacts: CalibrationArtifacts,
) -> str | None:
    """Why a question set is unusable, or ``None`` when it is fully supported."""
    codebook = build_codebook(compiled, questions)
    if codebook.size == 0 or len(codebook.target_classes) < 2:
        return f"degenerate_feasible_code:{codebook.signature}"
    model = artifacts.covariances.get(codebook.signature)
    if model is None:
        return f"unsupported_signature:{codebook.signature}"
    if model.dimension != codebook.dimension:
        return f"dimension_mismatch:{codebook.signature}"
    if not model.is_positive_definite:
        return f"non_positive_definite:{codebook.signature}"
    return None


def coverage_reason(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    options: InferenceOptions,
    selected: Sequence[QuestionSpec] | None = None,
) -> str | None:
    """Return why a record is outside QCV coverage, or ``None`` when it is covered.

    Coverage requires at least ``B - 1`` valid auxiliaries, a calibrated
    temperature for every question, and a positive-definite calibrated
    covariance for every *required* role signature. For the optimising selectors
    (greedy QCV and exhaustive search) the required signatures are those of all
    question sets with 1..B-1 auxiliaries, so the selection is well defined
    whichever path it takes; for the fixed, random and reliability controls only
    the signature of the selected set (``selected``) is required.
    """
    compiled = record.compiled
    if len(compiled.auxiliaries) < options.budget - 1:
        return "insufficient_valid_auxiliaries"
    for question in compiled.all_questions:
        if question.role.key not in artifacts.temperatures:
            return f"missing_temperature:{question.role.key}"
    if options.selection in OPTIMISING_SELECTIONS:
        if not options.strict_coverage:
            return None
        minimum = min(1, options.budget - 1)
        required = question_subsets(compiled, options.budget - 1, min_auxiliaries=minimum)
    else:
        if selected is None:
            raise ValueError("selected is required for non-optimising selection methods")
        required = iter([(compiled.target, *selected)])
    for questions in required:
        reason = _signature_reason(compiled, questions, artifacts)
        if reason is not None:
            return reason
    return None


@dataclass
class InferenceEngine:
    """Runs the inference protocol for a fixed set of artifacts and options.

    Selection depends only on the compiled structure and the calibration
    artifacts, so greedy / exact / fixed / reliability selections and coverage
    decisions are memoised per structure across records.
    """

    artifacts: CalibrationArtifacts
    options: InferenceOptions = field(default_factory=InferenceOptions)
    _selections: dict[str, SelectionResult | str] = field(default_factory=dict, repr=False)

    def _select(self, compiled: CompiledInstance) -> SelectionResult:
        options = self.options
        if options.selection == "qcv":
            return greedy_select(compiled, options.budget, self.artifacts, options.metric)
        if options.selection == "exact":
            return exact_select(
                compiled, options.budget, self.artifacts, options.metric, options.exact_subset_limit
            )
        if options.selection == "fixed":
            return fixed_select(compiled, options.budget)
        if options.selection == "random":
            return random_select(compiled, options.budget, options.random_seed)
        return reliability_select(compiled, options.budget, self.artifacts)

    def _covered_selection(self, record: ScoreRecord) -> SelectionResult | str:
        """Selected auxiliaries, or the reason the record falls back to target-only."""
        compiled = record.compiled
        options = self.options
        per_instance = options.selection == "random"
        if not per_instance and compiled.structure_key in self._selections:
            return self._selections[compiled.structure_key]
        outcome: SelectionResult | str
        try:
            if options.selection in OPTIMISING_SELECTIONS:
                outcome = coverage_reason(record, self.artifacts, options) or self._select(compiled)
            else:
                selection = self._select(compiled)
                outcome = (
                    coverage_reason(record, self.artifacts, options, selection.selected)
                    or selection
                )
        except KeyError as exc:
            outcome = f"selection_error:{exc.args[0] if exc.args else exc}"
        if not per_instance:
            self._selections[compiled.structure_key] = outcome
        return outcome

    def _fallback(self, record: ScoreRecord, target_only: str, reason: str) -> Prediction:
        options = self.options
        return Prediction(
            instance_id=record.instance_id,
            covered=False,
            target_only_answer=target_only,
            final_answer=target_only,
            selected_question_ids=(),
            target_margin=None,
            projection_gap=None,
            target_class_separation=None,
            reason=reason,
            budget=options.budget,
            selection=options.selection,
            metric=options.metric,
            measurements=((record.compiled.target.question_id, target_only),),
        )

    def infer(self, record: ScoreRecord) -> Prediction:
        """Predict the target answer of one record.

        Covered records are decoded with Eq. (7) from exactly ``B`` score blocks
        (target plus ``B - 1`` selected auxiliaries). Every other record uses the
        single target-only fallback ``argmax_y p_0[y]``.
        """
        target = record.compiled.target
        # Temperature scaling preserves the arg-max, so the fallback needs no calibration.
        raw = np.asarray(record.scores[target.question_id].mean_logprobs, dtype=np.float64)
        target_only = target.role.alphabet[int(np.argmax(raw))]

        outcome = self._covered_selection(record)
        if isinstance(outcome, str):
            return self._fallback(record, target_only, outcome)

        options = self.options
        questions = record.compiled.ordered_subset(
            [question.question_id for question in outcome.selected]
        )
        try:
            codebook = build_codebook(record.compiled, questions)
            covariance = self.artifacts.covariance_for(codebook.signature)
            whitening = whitening_for_mode(covariance, options.metric)
            probabilities = probabilities_for_record(record, self.artifacts, questions)
            observation = stack_observation(codebook.questions, probabilities)
            decoded = feasible_projection_decode(codebook, observation, whitening)
            separation = set_separation(record.compiled, questions, self.artifacts, options.metric)
        except (KeyError, ValueError, FloatingPointError) as exc:
            return self._fallback(
                record, target_only, f"inference_error:{type(exc).__name__}:{exc}"
            )

        return Prediction(
            instance_id=record.instance_id,
            covered=True,
            target_only_answer=target_only,
            final_answer=decoded.target_answer,
            selected_question_ids=tuple(question.question_id for question in questions[1:]),
            target_margin=decoded.target_margin,
            projection_gap=decoded.projection_gap,
            target_class_separation=separation,
            energies=decoded.target_energies,
            reason="qcv",
            budget=options.budget,
            selection=options.selection,
            metric=options.metric,
            measurements=tuple(
                (
                    question.question_id,
                    question.role.alphabet[int(np.argmax(probabilities[question.question_id]))],
                )
                for question in questions
            ),
        )


def infer_record(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    options: InferenceOptions,
) -> Prediction:
    """Convenience wrapper for a single record; use ``InferenceEngine`` for whole files."""
    return InferenceEngine(artifacts, options).infer(record)
