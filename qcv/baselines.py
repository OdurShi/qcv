"""Matched-budget baselines that can be computed from a QCV score file.

Only baselines that need nothing beyond the stored full-alphabet target scores
are provided. Baselines of the paper that require free-form generation or
search with the VLM (Self-Consistency, Best-of-N, beam search, MCTS,
Least-to-Most, visual question decomposition, implication projection) are not
reproducible from score files and are not implemented here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .temperature import calibrated_probabilities
from .types import CalibrationArtifacts, Prediction, ScoreRecord
from .utils import stable_hash

BASELINE_METHODS = ("target-only", "repeated-target")


def _target_distribution(record: ScoreRecord, artifacts: CalibrationArtifacts) -> np.ndarray:
    target = record.compiled.target
    scores = record.scores[target.question_id].mean_logprobs
    temperature = artifacts.temperatures.get(target.role.key, 1.0)
    return calibrated_probabilities(scores, temperature)


def target_only_prediction(record: ScoreRecord, artifacts: CalibrationArtifacts) -> Prediction:
    """Target-only decoding ``argmax_y p_0[y]``: one evaluation of the target question."""
    target = record.compiled.target
    distribution = _target_distribution(record, artifacts)
    answer = target.role.alphabet[int(np.argmax(distribution))]
    return Prediction(
        instance_id=record.instance_id,
        covered=True,
        target_only_answer=answer,
        final_answer=answer,
        selected_question_ids=(),
        target_margin=None,
        projection_gap=None,
        target_class_separation=None,
        reason="target_only",
        budget=1,
        selection="target-only",
        measurements=((target.question_id, answer),),
    )


def repeated_target_prediction(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    budget: int,
    seed: int = 0,
) -> Prediction:
    """Repeated target: ``B`` stochastic evaluations of the target and a plurality vote.

    Each evaluation is an independent draw from the calibrated target
    distribution ``p_0`` (seeded per instance). Vote ties go to the answer with
    the larger ``p_0`` and then to canonical alphabet order. This is a score-file
    stand-in for spending the whole budget on the target question; it is not the
    chain-of-thought Self-Consistency baseline of the paper.
    """
    if budget < 1:
        raise ValueError("budget must be at least one")
    target = record.compiled.target
    distribution = _target_distribution(record, artifacts)
    alphabet = target.role.alphabet
    instance_seed = int(stable_hash([str(seed), record.instance_id, "repeat"], length=16), 16)
    generator = np.random.default_rng(instance_seed)
    draws = generator.choice(len(alphabet), size=budget, p=distribution)
    votes = Counter(int(index) for index in draws)
    winner = min(votes, key=lambda index: (-votes[index], -distribution[index], index))
    return Prediction(
        instance_id=record.instance_id,
        covered=True,
        target_only_answer=alphabet[int(np.argmax(distribution))],
        final_answer=alphabet[winner],
        selected_question_ids=(),
        target_margin=None,
        projection_gap=None,
        target_class_separation=None,
        reason="repeated_target",
        budget=budget,
        selection="repeated-target",
        measurements=tuple((target.question_id, alphabet[int(index)]) for index in draws),
    )


def baseline_prediction(
    record: ScoreRecord,
    artifacts: CalibrationArtifacts,
    method: str,
    budget: int = 5,
    seed: int = 0,
) -> Prediction:
    if method == "target-only":
        return target_only_prediction(record, artifacts)
    if method == "repeated-target":
        return repeated_target_prediction(record, artifacts, budget, seed)
    raise ValueError(f"Unknown baseline {method!r}; expected one of {BASELINE_METHODS}")
