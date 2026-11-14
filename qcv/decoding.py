"""Feasible projection decoding (Eqs. 6 and 7)."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .codebook import Codebook
from .utils import first_argmin

MAX_UNCONSTRAINED_TUPLES = 250_000
"""Largest answer product for which the unconstrained minimum energy is evaluated."""
TIE_TOLERANCE = 1e-12
"""Relative tolerance below which two energies count as tied."""

_GRID_CACHE: OrderedDict[tuple[str, bytes], tuple[np.ndarray, np.ndarray]] = OrderedDict()
_GRID_CACHE_SIZE = 32


@dataclass(frozen=True)
class DecodeResult:
    target_answer: str
    target_energies: dict[str, float]
    """Class energy ``min_{c in C(y)} E(c; z)`` for every feasible target answer ``y``."""
    target_margin: float
    """Runner-up class energy minus winning class energy (``inf`` for a single class)."""
    projection_gap: float | None
    """Winning feasible energy minus the minimum energy over all one-hot tuples."""
    best_codeword_index: int


def _first_minimum(values: np.ndarray) -> int:
    """Index of the smallest energy; energies equal up to rounding keep canonical order."""
    tolerance = TIE_TOLERANCE * max(1.0, float(np.max(np.abs(values))))
    return first_argmin(values.tolist(), atol=tolerance)


def codeword_energies(
    codewords: np.ndarray, observation: np.ndarray, whitening: np.ndarray
) -> np.ndarray:
    """``E(c; z) = ||W (z - c)||_2^2`` for every row ``c`` of ``codewords`` (Eq. 6)."""
    residuals = observation[None, :] - codewords
    whitened = residuals @ np.asarray(whitening, dtype=np.float64).T
    return np.sum(whitened * whitened, axis=1)


def _quadratic_grid(codebook: Codebook, whitening: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``M = W^T W`` and the grid of ``c^T M c`` over the full answer product (cached)."""
    key = (codebook.structure_key, hashlib.blake2b(whitening.tobytes(), digest_size=16).digest())
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        _GRID_CACHE.move_to_end(key)
        return cached
    metric = whitening.T @ whitening
    sizes = codebook.block_sizes
    blocks = [codebook.block_slices[question.question_id] for question in codebook.questions]
    grid = np.zeros(sizes, dtype=np.float64)
    for j, block_j in enumerate(blocks):
        shape_j = [1] * len(blocks)
        shape_j[j] = sizes[j]
        grid = grid + np.diag(metric)[block_j].reshape(shape_j)
        for k in range(j + 1, len(blocks)):
            shape_jk = [1] * len(blocks)
            shape_jk[j] = sizes[j]
            shape_jk[k] = sizes[k]
            grid = grid + 2.0 * metric[block_j, blocks[k]].reshape(shape_jk)
    _GRID_CACHE[key] = (metric, grid)
    if len(_GRID_CACHE) > _GRID_CACHE_SIZE:
        _GRID_CACHE.popitem(last=False)
    return metric, grid


def unconstrained_minimum_energy(
    codebook: Codebook, observation: np.ndarray, whitening: np.ndarray
) -> float | None:
    """Minimum of Eq. (6) over the full answer product, ignoring the predicates.

    With ``M = W^T W`` and one-hot blocks, ``E(c) = z^T M z - 2 sum_k (M z)[i_k]
    + sum_{j,k} M[i_j, i_k]``, so the minimum is taken over a dense grid with
    one axis per question without materialising any codeword. Returns ``None``
    when the product exceeds ``MAX_UNCONSTRAINED_TUPLES``.
    """
    sizes = codebook.block_sizes
    if float(np.prod(sizes, dtype=np.float64)) > MAX_UNCONSTRAINED_TUPLES:
        return None
    whitening = np.ascontiguousarray(whitening, dtype=np.float64)
    metric, grid = _quadratic_grid(codebook, whitening)
    linear = metric @ observation
    for axis, question in enumerate(codebook.questions):
        shape = [1] * len(sizes)
        shape[axis] = sizes[axis]
        grid = grid - 2.0 * linear[codebook.block_slices[question.question_id]].reshape(shape)
    return float(observation @ linear + np.min(grid))


def feasible_projection_decode(
    codebook: Codebook,
    observation: np.ndarray,
    whitening: np.ndarray,
) -> DecodeResult:
    """Project ``z`` onto the feasible code and read off the target class (Eq. 7).

    ``y_hat = argmin_y min_{c in C(y)} ||W (z - c)||_2^2``. Ties between classes,
    and between codewords inside a class, go to the first entry in canonical
    alphabet / answer-product order.
    """
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (codebook.dimension,):
        raise ValueError("Observation dimension does not match codebook")
    if codebook.size == 0:
        raise ValueError("Cannot decode an empty codebook")

    energies = codeword_energies(codebook.codewords, observation, whitening)
    target_energies: dict[str, float] = {}
    best_rows: dict[str, int] = {}
    answers = [answer for answer in codebook.target_alphabet if answer in codebook.target_classes]
    for answer in answers:
        rows = codebook.target_classes[answer]
        row = int(rows[_first_minimum(energies[rows])])
        target_energies[answer] = float(energies[row])
        best_rows[answer] = row

    winner = answers[_first_minimum(np.asarray([target_energies[a] for a in answers]))]
    competitors = [target_energies[answer] for answer in answers if answer != winner]
    margin = float("inf")
    if competitors:
        margin = max(float(min(competitors) - target_energies[winner]), 0.0)

    unconstrained = unconstrained_minimum_energy(codebook, observation, whitening)
    projection_gap = None
    if unconstrained is not None:
        projection_gap = max(float(target_energies[winner] - unconstrained), 0.0)

    return DecodeResult(
        target_answer=winner,
        target_energies=target_energies,
        target_margin=margin,
        projection_gap=projection_gap,
        best_codeword_index=best_rows[winner],
    )
