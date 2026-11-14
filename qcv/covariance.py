"""Residual covariance, identity-target shrinkage and whitening (Eq. 3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import CovarianceModel

EIGENVALUE_TOLERANCE = 1e-12


def ledoit_wolf_shrinkage(centered: np.ndarray) -> float:
    """Ledoit-Wolf (2004) shrinkage intensity ``alpha_LW`` towards ``mu * I``.

    ``centered`` holds one mean-centred residual per row. With ``S = X^T X / N``
    and ``mu = tr(S) / D`` the estimator is ``min(b^2, d^2) / d^2`` where
    ``d^2 = ||S - mu I||_F^2 / D`` and
    ``b^2 = (1 / N^2) sum_n ||x_n x_n^T - S||_F^2 / D``.
    """
    sample_count, dimension = centered.shape
    covariance = (centered.T @ centered) / float(sample_count)
    mu = float(np.trace(covariance) / dimension)
    delta = float(np.sum((covariance - mu * np.eye(dimension)) ** 2) / dimension)
    if delta <= 0.0:
        return 0.0
    squared_norms = np.sum(centered * centered, axis=1) ** 2
    beta = float(np.mean(squared_norms) - np.sum(covariance**2)) / float(dimension * sample_count)
    return float(np.clip(beta / delta, 0.0, 1.0))


def inverse_sqrt(matrix: np.ndarray, tolerance: float = EIGENVALUE_TOLERANCE) -> np.ndarray:
    """Symmetric inverse square root of a symmetric positive-definite matrix."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum = float(np.min(eigenvalues))
    if minimum <= tolerance:
        raise ValueError(f"Matrix is not positive definite: minimum eigenvalue={minimum:.3e}")
    root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return 0.5 * (root + root.T)


def fit_residual_covariance(
    signature: str,
    role_keys: Sequence[str],
    residual_rows: Sequence[Sequence[float]],
) -> CovarianceModel:
    """Fit the shrunk residual covariance and its whitening transform (Eq. 3).

    ``S`` is the centred covariance of the residuals ``r = z - c`` of one role
    signature, ``alpha = max(alpha_LW, 1/N)``, ``mu = tr(S)/D``,
    ``Sigma = (1 - alpha) S + alpha mu I`` and ``W = Sigma^{-1/2}``.

    Raises ``ValueError`` when the residual variance is zero (``mu = 0``); such
    signatures are outside QCV coverage and fall back to target-only decoding.
    """
    residuals = np.asarray(residual_rows, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residual_rows must form a two-dimensional array")
    sample_count, dimension = residuals.shape
    if sample_count < 2:
        raise ValueError("At least two complete residual observations are required")
    if dimension < 1:
        raise ValueError("Residual dimension must be positive")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Residual covariance received non-finite values")

    centered = residuals - np.mean(residuals, axis=0, keepdims=True)
    empirical = (centered.T @ centered) / float(sample_count)
    mu = float(np.trace(empirical) / dimension)
    if not np.isfinite(mu) or mu <= EIGENVALUE_TOLERANCE:
        raise ValueError("Residual variance is zero (below numerical tolerance) or non-finite")

    alpha_lw = ledoit_wolf_shrinkage(centered)
    alpha = min(max(alpha_lw, 1.0 / sample_count), 1.0)
    covariance = (1.0 - alpha) * empirical + alpha * mu * np.eye(dimension)
    covariance = 0.5 * (covariance + covariance.T)
    whitening = inverse_sqrt(covariance)

    return CovarianceModel(
        signature=signature,
        role_keys=tuple(str(item) for item in role_keys),
        dimension=dimension,
        sample_count=sample_count,
        alpha_lw=alpha_lw,
        alpha=alpha,
        mu=mu,
        covariance=covariance,
        whitening=whitening,
    )
