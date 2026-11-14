import numpy as np
import pytest

from qcv.temperature import _objective, calibrated_probabilities, fit_temperature


def _sample(generator, temperature, examples=20_000, classes=5):
    """Scores whose correctly calibrated temperature is ``temperature``."""
    logits = generator.normal(scale=2.0, size=(examples, classes))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    gold = np.array([generator.choice(classes, p=row) for row in probabilities])
    return temperature * logits, gold


@pytest.mark.parametrize("temperature", [0.4, 1.0, 2.5])
def test_recovers_known_temperature(temperature):
    scores, gold = _sample(np.random.default_rng(0), temperature)
    fit = fit_temperature(scores, gold)
    assert fit.converged
    assert fit.temperature == pytest.approx(temperature, rel=0.05)


def test_temperature_is_equivariant_to_score_scale():
    scores, gold = _sample(np.random.default_rng(1), 1.0, examples=2000)
    base = fit_temperature(scores, gold).temperature
    assert fit_temperature(3.0 * scores, gold).temperature == pytest.approx(3.0 * base, rel=1e-4)


def test_objective_value_and_gradient():
    scores = np.array([[0.0, 1.0, 3.0], [2.0, 0.5, -1.0], [0.3, 0.2, 0.1]])
    gold = np.array([2, 1, 0])
    point = np.array([0.3])
    value, gradient = _objective(point, scores, gold)
    scaled = scores / np.exp(0.3)
    expected = np.mean(np.log(np.exp(scaled).sum(axis=1)) - scaled[np.arange(3), gold])
    assert value == pytest.approx(expected)
    step = 1e-6
    numeric = (
        _objective(point + step, scores, gold)[0] - _objective(point - step, scores, gold)[0]
    ) / (2 * step)
    assert gradient[0] == pytest.approx(numeric, rel=1e-6)


def test_fitted_temperature_minimises_the_nll():
    scores, gold = _sample(np.random.default_rng(2), 1.7, examples=3000)
    fit = fit_temperature(scores, gold)
    for factor in (0.8, 0.95, 1.05, 1.25):
        other, _ = _objective(np.array([np.log(fit.temperature * factor)]), scores, gold)
        assert other > fit.nll


def test_calibrated_probabilities_and_input_validation():
    probabilities = calibrated_probabilities([0.0, np.log(3.0)], 1.0)
    assert np.allclose(probabilities, [0.25, 0.75])
    assert calibrated_probabilities([1.0, 2.0, 3.0], 0.7).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 3)), [0, 3])
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((0, 3)), [])
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 3)), [0])
