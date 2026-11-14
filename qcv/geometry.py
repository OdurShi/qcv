"""Whitened residual geometry: metrics, target-class separation and margins (Eqs. 4 and 8)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .codebook import Codebook
from .types import METRIC_MODES, CovarianceModel, QuestionSpec


def whitening_for_mode(model: CovarianceModel, mode: str) -> np.ndarray:
    """Transform ``W`` applied to residuals before taking Euclidean norms.

    ``full``      ``Sigma^{-1/2}`` of the shrunk covariance (QCV, Eq. 3).
    ``diagonal``  ``diag(Sigma)^{-1/2}``: per-coordinate reliabilities only (Diagonal-QCV).
    ``identity``  no reweighting (plain Euclidean projection).
    """
    if mode == "full":
        return np.asarray(model.whitening, dtype=np.float64)
    if mode == "diagonal":
        diagonal = np.diag(np.asarray(model.covariance, dtype=np.float64))
        if np.any(diagonal <= 0.0):
            raise ValueError("Diagonal covariance contains a non-positive entry")
        return np.diag(1.0 / np.sqrt(diagonal))
    if mode == "identity":
        return np.eye(model.dimension, dtype=np.float64)
    raise ValueError(f"Unknown metric mode {mode!r}; expected one of {METRIC_MODES}")


def _minimum_cross_distance(left: np.ndarray, right: np.ndarray, chunk_size: int = 2048) -> float:
    best = math.inf
    for start in range(0, left.shape[0], chunk_size):
        chunk = left[start : start + chunk_size]
        differences = chunk[:, None, :] - right[None, :, :]
        best = min(best, float(np.min(np.sum(differences * differences, axis=2))))
        if best <= 0.0:
            return 0.0
    return math.sqrt(best)


def target_class_separation(codebook: Codebook, whitening: np.ndarray) -> float:
    """Minimum whitened distance between distinct target-answer classes (Eq. 4).

    ``d_tar(Q̄) = min_{y != y'} min_{c in C(y), c' in C(y')} ||W (c - c')||_2``.
    Returns 0 when fewer than two target classes are feasible.
    """
    if codebook.size == 0:
        raise ValueError("Cannot compute target separation for an empty codebook")
    answers = [answer for answer in codebook.target_alphabet if answer in codebook.target_classes]
    if len(answers) < 2:
        return 0.0
    transformed = codebook.codewords @ np.asarray(whitening, dtype=np.float64).T
    best = math.inf
    for left_index, left_answer in enumerate(answers):
        left = transformed[codebook.target_classes[left_answer]]
        for right_answer in answers[left_index + 1 :]:
            right = transformed[codebook.target_classes[right_answer]]
            best = min(best, _minimum_cross_distance(left, right))
            if best <= 0.0:
                return 0.0
    return float(best)


def whitened_residual_radius(
    observation: np.ndarray, codeword: np.ndarray, whitening: np.ndarray
) -> float:
    """``r = ||W (z - c)||_2``; with the true codeword this is ``||W eps||_2`` of Eq. (8)."""
    residual = np.asarray(observation, dtype=np.float64) - np.asarray(codeword, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(whitening, dtype=np.float64) @ residual))


def normalized_margin(residual_radius: float, separation: float) -> float:
    """``kappa = 2 r / d_tar``. Eq. (8) guarantees a correct target whenever ``kappa < 1``."""
    if separation <= 0.0:
        return math.inf
    return 2.0 * residual_radius / separation


def stack_observation(
    questions: Sequence[QuestionSpec], probabilities: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Stacked observation ``z_Q̄ = col_k p_k`` in codebook block order (Eq. 2)."""
    blocks: list[np.ndarray] = []
    for question in questions:
        try:
            block = np.asarray(probabilities[question.question_id], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(f"Missing probability block for {question.question_id}") from exc
        if block.shape != (len(question.role.alphabet),):
            raise ValueError(f"Probability block shape mismatch for {question.question_id}")
        if float(block.min()) < 0.0 or abs(float(block.sum()) - 1.0) > 1e-8:
            raise ValueError(f"Invalid probability distribution for {question.question_id}")
        blocks.append(block)
    return np.concatenate(blocks, axis=0)
