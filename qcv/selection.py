"""Fixed-budget auxiliary selection (Eq. 5) and the selection controls of the paper.

Every selector returns exactly ``B - 1`` auxiliaries in compiler order and uses
only the compiled instance and the frozen calibration artifacts: no score of the
test instance is read, so selection can run before the VLM is queried.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .codebook import Codebook, build_codebook
from .geometry import target_class_separation, whitening_for_mode
from .types import CalibrationArtifacts, CompiledInstance, CovarianceModel, QuestionSpec
from .utils import stable_hash

SELECTION_METHODS = ("qcv", "exact", "fixed", "random", "reliability")
TIE_TOLERANCE = 1e-12
"""Relative tolerance below which two separations count as tied."""

_SEPARATION_CACHE: dict[tuple[str, str, str], float] = {}


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[QuestionSpec, ...]
    separation: float
    evaluated_subsets: int


def _compiler_order(question: QuestionSpec) -> tuple[int, str, str]:
    return (question.template_index, question.text, question.question_id)


def _improves(score: float, best: float) -> bool:
    """Strict improvement beyond numerical noise; ties keep the earlier candidate."""
    if best == -math.inf:
        return True
    return score > best + TIE_TOLERANCE * max(1.0, abs(best))


def _require_pool(compiled: CompiledInstance, budget: int) -> int:
    if budget < 1:
        raise ValueError("budget must be at least one")
    required = budget - 1
    if len(compiled.auxiliaries) < required:
        raise KeyError("The compiled pool is smaller than the requested budget")
    return required


def supported_codebook(
    compiled: CompiledInstance,
    questions: Sequence[QuestionSpec],
    artifacts: CalibrationArtifacts,
) -> tuple[Codebook, CovarianceModel]:
    """Codebook and calibrated covariance of a question set, or ``KeyError`` if unsupported."""
    codebook = build_codebook(compiled, questions)
    if codebook.size == 0 or len(codebook.target_classes) < 2:
        raise KeyError("The selected set has no usable feasible target classes")
    covariance = artifacts.covariance_for(codebook.signature)
    if covariance.dimension != codebook.dimension:
        raise KeyError("Covariance dimension and codebook dimension differ")
    return codebook, covariance


def set_separation(
    compiled: CompiledInstance,
    questions: Sequence[QuestionSpec],
    artifacts: CalibrationArtifacts,
    metric_mode: str = "full",
) -> float:
    """``d_tar({q0} ∪ Q)`` in the calibrated geometry of that question set (Eq. 4)."""
    codebook, covariance = supported_codebook(compiled, questions, artifacts)
    key = (codebook.structure_key, covariance.fingerprint, metric_mode)
    cached = _SEPARATION_CACHE.get(key)
    if cached is None:
        cached = target_class_separation(codebook, whitening_for_mode(covariance, metric_mode))
        _SEPARATION_CACHE[key] = cached
    return cached


def greedy_select(
    compiled: CompiledInstance,
    budget: int,
    artifacts: CalibrationArtifacts,
    metric_mode: str = "full",
) -> SelectionResult:
    """Deterministic greedy maximisation of Eq. (5) for exactly ``B - 1`` steps.

    Each step adds the auxiliary that maximises ``d_tar`` of the enlarged set,
    evaluated with that set's own whitening transform. Ties are broken by
    compiler template index (candidates are scanned in compiler order and only
    a strict improvement replaces the incumbent).
    """
    required = _require_pool(compiled, budget)
    selected: list[QuestionSpec] = []
    evaluated = 0
    separation = 0.0
    for _ in range(required):
        best_question: QuestionSpec | None = None
        best_score = -math.inf
        for candidate in compiled.auxiliaries:
            if candidate in selected:
                continue
            try:
                score = set_separation(
                    compiled, (compiled.target, *selected, candidate), artifacts, metric_mode
                )
            except KeyError:
                continue
            evaluated += 1
            if _improves(score, best_score):
                best_question = candidate
                best_score = score
        if best_question is None:
            raise KeyError("No supported candidate remains during greedy selection")
        selected.append(best_question)
        separation = best_score
    if not selected:
        separation = set_separation(compiled, (compiled.target,), artifacts, metric_mode)
    return SelectionResult(
        tuple(sorted(selected, key=_compiler_order)), float(separation), evaluated
    )


def exact_select(
    compiled: CompiledInstance,
    budget: int,
    artifacts: CalibrationArtifacts,
    metric_mode: str = "full",
    subset_limit: int = 10_000,
) -> SelectionResult:
    """Exhaustive solution of Eq. (5); used to audit the greedy selector.

    Ties go to the lexicographically first subset in compiler order. Pools with
    more than ``subset_limit`` candidate subsets fall back to greedy selection.
    """
    required = _require_pool(compiled, budget)
    if math.comb(len(compiled.auxiliaries), required) > subset_limit:
        return greedy_select(compiled, budget, artifacts, metric_mode)
    best_subset: tuple[QuestionSpec, ...] | None = None
    best_score = -math.inf
    evaluated = 0
    for subset in itertools.combinations(compiled.auxiliaries, required):
        try:
            score = set_separation(compiled, (compiled.target, *subset), artifacts, metric_mode)
        except KeyError:
            continue
        evaluated += 1
        if _improves(score, best_score):
            best_subset = subset
            best_score = score
    if best_subset is None:
        raise KeyError("No supported exact subset exists")
    return SelectionResult(best_subset, float(best_score), evaluated)


def fixed_select(compiled: CompiledInstance, budget: int) -> SelectionResult:
    """Fixed related probes: the first ``B - 1`` auxiliaries in compiler order."""
    required = _require_pool(compiled, budget)
    return SelectionResult(compiled.auxiliaries[:required], float("nan"), 0)


def reliability_select(
    compiled: CompiledInstance,
    budget: int,
    artifacts: CalibrationArtifacts,
) -> SelectionResult:
    """Reliability-QCV: the ``B - 1`` auxiliaries whose roles are most accurate on calibration."""
    required = _require_pool(compiled, budget)
    ranked = sorted(
        compiled.auxiliaries,
        key=lambda question: (
            -artifacts.role_reliability(question.role),
            *_compiler_order(question),
        ),
    )
    return SelectionResult(tuple(sorted(ranked[:required], key=_compiler_order)), float("nan"), 0)


def random_select(
    compiled: CompiledInstance,
    budget: int,
    seed: int,
) -> SelectionResult:
    """Random-QCV: a uniformly random subset, seeded per instance for reproducibility."""
    required = _require_pool(compiled, budget)
    instance_seed = int(stable_hash([str(seed), compiled.instance_id], length=16), 16)
    generator = random.Random(instance_seed)
    selected = generator.sample(list(compiled.auxiliaries), required)
    return SelectionResult(tuple(sorted(selected, key=_compiler_order)), float("nan"), 0)
