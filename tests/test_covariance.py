import numpy as np
import pytest

from qcv.covariance import fit_residual_covariance, inverse_sqrt, ledoit_wolf_shrinkage


def _isotropic_design(dimension: int) -> np.ndarray:
    """Zero-mean rows with X^T X / N = I exactly: +/- sqrt(D) e_i."""
    scale = np.sqrt(dimension)
    return np.concatenate([scale * np.eye(dimension), -scale * np.eye(dimension)])


def test_identity_covariance_gives_identity_whitening():
    model = fit_residual_covariance("sig", ["a"], _isotropic_design(4))
    assert model.mu == pytest.approx(1.0)
    assert model.alpha_lw == 0.0
    assert model.alpha == pytest.approx(1.0 / 8.0)  # the deterministic 1/N floor
    assert np.allclose(model.covariance, np.eye(4), atol=1e-12)
    assert np.allclose(model.whitening, np.eye(4), atol=1e-12)


def test_hand_computed_shrinkage():
    # S = diag(0.5, 2), mu = 1.25, d^2 = 0.5625, b^2 = (8.5 - 4.25) / (2 * 4) = 0.53125.
    rows = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    assert ledoit_wolf_shrinkage(rows) == pytest.approx(17.0 / 18.0)
    model = fit_residual_covariance("sig", ["a"], rows)
    alpha = 17.0 / 18.0
    expected = (1 - alpha) * np.diag([0.5, 2.0]) + alpha * 1.25 * np.eye(2)
    assert model.alpha == pytest.approx(alpha)
    assert np.allclose(model.covariance, expected)
    assert np.allclose(model.whitening, np.diag(1.0 / np.sqrt(np.diag(expected))))


def test_covariance_is_centred():
    generator = np.random.default_rng(0)
    rows = generator.normal(size=(200, 3))
    shifted = rows + np.array([5.0, -3.0, 0.5])
    first = fit_residual_covariance("sig", ["a"], rows)
    second = fit_residual_covariance("sig", ["a"], shifted)
    assert np.allclose(first.covariance, second.covariance)


def test_whitening_reconstruction_and_symmetry():
    generator = np.random.default_rng(7)
    residuals = generator.normal(size=(128, 8))
    residuals[:, 1] = residuals[:, 0] * 0.9 + generator.normal(scale=0.05, size=128)
    model = fit_residual_covariance("sig", ["a", "b"], residuals)
    assert np.min(np.linalg.eigvalsh(model.covariance)) > 0
    assert np.allclose(model.whitening, model.whitening.T)
    assert np.min(np.linalg.eigvalsh(model.whitening)) > 0
    identity = model.whitening @ model.covariance @ model.whitening
    assert np.max(np.abs(identity - np.eye(8))) < 1e-9
    assert np.allclose(model.whitening @ model.whitening, np.linalg.inv(model.covariance))
    assert model.alpha >= 1 / 128


def test_shrinkage_removes_simplex_null_directions():
    generator = np.random.default_rng(1)
    probabilities = generator.dirichlet(np.ones(4), size=300)
    residuals = probabilities - np.eye(4)[generator.integers(0, 4, size=300)]
    centered = residuals - residuals.mean(axis=0)
    empirical = centered.T @ centered / 300
    assert np.min(np.linalg.eigvalsh(empirical)) < 1e-12  # rows sum to zero: singular S
    model = fit_residual_covariance("sig", ["a"], residuals)
    assert np.min(np.linalg.eigvalsh(model.covariance)) == pytest.approx(
        model.alpha * model.mu, rel=1e-6
    )


def test_matches_scikit_learn_ledoit_wolf():
    sklearn_covariance = pytest.importorskip("sklearn.covariance")
    generator = np.random.default_rng(3)
    mixing = generator.normal(size=(6, 6))
    rows = generator.normal(size=(40, 6)) @ mixing
    reference, shrinkage = sklearn_covariance.ledoit_wolf(rows)
    model = fit_residual_covariance("sig", ["a"], rows)
    assert shrinkage > 1 / 40
    assert model.alpha_lw == pytest.approx(shrinkage, rel=1e-10)
    assert np.allclose(model.covariance, reference, atol=1e-10)


def test_zero_variance_and_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="zero"):
        fit_residual_covariance("sig", ["a"], np.ones((5, 3)))
    with pytest.raises(ValueError):
        fit_residual_covariance("sig", ["a"], np.ones((1, 3)))
    with pytest.raises(ValueError):
        fit_residual_covariance("sig", ["a"], [[np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        inverse_sqrt(np.diag([1.0, 0.0]))


def test_inverse_sqrt_of_diagonal_matrix():
    assert np.allclose(inverse_sqrt(np.diag([4.0, 9.0])), np.diag([0.5, 1.0 / 3.0]))
