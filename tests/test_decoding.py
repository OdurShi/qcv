import itertools

import numpy as np
import pytest

from qcv.codebook import build_codebook
from qcv.decoding import (
    codeword_energies,
    feasible_projection_decode,
    unconstrained_minimum_energy,
)
from qcv.geometry import target_class_separation


def _random_spd(generator, dimension):
    mixing = generator.normal(size=(dimension, dimension))
    return mixing @ mixing.T / dimension + 0.3 * np.eye(dimension)


def _whitening(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def test_energy_is_the_squared_whitened_distance():
    whitening = np.array([[2.0, 0.0], [1.0, 3.0]])
    codewords = np.array([[1.0, 0.0], [0.0, 1.0]])
    observation = np.array([0.75, 0.25])
    # W (z - c1) = W (-0.25, 0.25) = (-0.5, 0.5);  W (z - c2) = W (0.75, -0.75) = (1.5, -1.5).
    assert np.allclose(codeword_energies(codewords, observation, whitening), [0.5, 4.5])


def test_no_auxiliaries_reduces_to_target_argmax(compile_question):
    compiled = compile_question("clevr", "How many red cubes are there?")
    codebook = build_codebook(compiled, (compiled.target,))
    generator = np.random.default_rng(0)
    for _ in range(200):
        distribution = generator.dirichlet(np.ones(codebook.dimension))
        decoded = feasible_projection_decode(codebook, distribution, np.eye(codebook.dimension))
        assert decoded.target_answer == codebook.target_alphabet[int(np.argmax(distribution))]


def test_binary_target_is_argmax_under_any_positive_definite_metric(compile_question):
    compiled = compile_question("clevr", "Are there any red cubes?")
    codebook = build_codebook(compiled, (compiled.target,))
    generator = np.random.default_rng(1)
    for _ in range(100):
        probability = generator.uniform(0.01, 0.99)
        whitening = _whitening(_random_spd(generator, 2))
        decoded = feasible_projection_decode(
            codebook, np.array([probability, 1 - probability]), whitening
        )
        assert decoded.target_answer == ("yes" if probability > 0.5 else "no")


def test_isotropic_covariance_makes_whitening_a_no_op(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    generator = np.random.default_rng(2)
    identity = np.eye(codebook.dimension)
    isotropic = _whitening(0.37 * identity)
    for _ in range(100):
        observation = np.concatenate([generator.dirichlet(np.ones(size)) for size in (5, 3, 3, 5)])
        plain = feasible_projection_decode(codebook, observation, identity)
        scaled = feasible_projection_decode(codebook, observation, isotropic)
        assert plain.target_answer == scaled.target_answer
        assert plain.best_codeword_index == scaled.best_codeword_index
        for answer, energy in plain.target_energies.items():
            assert scaled.target_energies[answer] == pytest.approx(energy / 0.37)


def test_decoded_codeword_is_feasible_and_carries_the_target(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    generator = np.random.default_rng(3)
    whitening = _whitening(_random_spd(generator, codebook.dimension))
    for _ in range(100):
        observation = np.concatenate([generator.dirichlet(np.ones(size)) for size in (5, 3, 3, 5)])
        decoded = feasible_projection_decode(codebook, observation, whitening)
        assignment = codebook.assignment(decoded.best_codeword_index)
        assert assignment["d"] == decoded.target_answer
        assert int(assignment["d"]) == int(assignment["b"]) - int(assignment["m"])
        assert int(assignment["t"]) == int(assignment["b"]) + int(assignment["m"])
        energies = codeword_energies(codebook.codewords, observation, whitening)
        assert decoded.target_energies[decoded.target_answer] == pytest.approx(energies.min())
        assert decoded.target_margin >= 0.0
        assert decoded.projection_gap >= 0.0


def test_ties_go_to_canonical_alphabet_order(figure3_instance):
    codebook = build_codebook(figure3_instance, (figure3_instance.target,))
    decoded = feasible_projection_decode(codebook, np.full(5, 0.2), np.eye(5))
    assert decoded.target_answer == codebook.target_alphabet[0]
    assert decoded.target_margin == 0.0


def test_constraints_overrule_a_wrong_target(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions[:3])
    observation = np.concatenate(
        [
            [0.05, 0.05, 0.1, 0.45, 0.35],  # d: prefers 1 over the true 2
            [0.02, 0.03, 0.95],  # b = 2
            [0.95, 0.03, 0.02],  # m = 0
        ]
    )
    decoded = feasible_projection_decode(codebook, observation, np.eye(11))
    assert decoded.target_answer == "2"
    assert codebook.assignment(decoded.best_codeword_index) == {"d": "2", "b": "2", "m": "0"}
    # Hand check of the two leading class energies under the identity metric.
    true = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    assert decoded.target_energies["2"] == pytest.approx(np.sum((observation - true) ** 2))
    assert decoded.target_margin > 0.0


def test_deterministic_correction_condition(figure3_instance):
    """Eq. (8): ||W eps|| < d_tar / 2 implies the decoded target is the true one."""
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    generator = np.random.default_rng(4)
    for _ in range(300):
        covariance = _random_spd(generator, codebook.dimension)
        whitening = _whitening(covariance)
        separation = target_class_separation(codebook, whitening)
        row = int(generator.integers(0, codebook.size))
        direction = generator.normal(size=codebook.dimension)
        direction /= np.linalg.norm(whitening @ direction)
        noise = generator.uniform(0.0, 0.999) * 0.5 * separation * direction
        decoded = feasible_projection_decode(codebook, codebook.codewords[row] + noise, whitening)
        assert decoded.target_answer == codebook.assignment(row)["d"]


def test_correction_condition_is_tight(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    identity = np.eye(codebook.dimension)
    separation = target_class_separation(codebook, identity)
    first = codebook.class_codewords("0")
    second = codebook.class_codewords("1")
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    assert distances[i, j] == pytest.approx(separation)
    beyond = first[i] + 0.51 * (second[j] - first[i])  # ||eps|| = 0.51 d_tar > d_tar / 2
    assert feasible_projection_decode(codebook, beyond, identity).target_answer == "1"


def test_unconstrained_minimum_matches_brute_force(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions[:3])
    generator = np.random.default_rng(5)
    whitening = _whitening(_random_spd(generator, codebook.dimension))
    observation = np.concatenate([generator.dirichlet(np.ones(size)) for size in (5, 3, 3)])
    product = []
    for indices in itertools.product(range(5), range(3), range(3)):
        vector = np.zeros(11)
        vector[[indices[0], 5 + indices[1], 8 + indices[2]]] = 1.0
        product.append(vector)
    brute = codeword_energies(np.array(product), observation, whitening).min()
    assert unconstrained_minimum_energy(codebook, observation, whitening) == pytest.approx(brute)


def test_projection_gap_is_zero_iff_the_nearest_tuple_is_feasible(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions[:3])
    feasible = codebook.encode_assignment({"d": "1", "b": "2", "m": "1"})
    infeasible = codebook.encode_assignment({"d": "2", "b": "2", "m": "1"})
    identity = np.eye(11)
    assert feasible_projection_decode(codebook, feasible, identity).projection_gap == 0.0
    assert feasible_projection_decode(codebook, infeasible, identity).projection_gap > 0.0


def test_whitening_beats_euclidean_projection_under_correlated_gaussian_noise(figure3_instance):
    """With z = c + eps, eps ~ N(0, Sigma), projection in the Sigma^{-1/2} geometry is the
    maximum-likelihood decoder; it must beat the identity metric when Sigma is anisotropic."""
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    generator = np.random.default_rng(6)
    dimension = codebook.dimension
    shared = generator.normal(size=dimension)
    shared /= np.linalg.norm(shared)
    covariance = 0.03 * np.eye(dimension) + 40.0 * np.outer(shared, shared)
    whitening = _whitening(covariance)
    cholesky = np.linalg.cholesky(covariance)
    correct = {"full": 0, "identity": 0}
    trials = 1500
    for _ in range(trials):
        row = int(generator.integers(0, codebook.size))
        observation = codebook.codewords[row] + cholesky @ generator.normal(size=dimension)
        truth = codebook.assignment(row)["d"]
        correct["full"] += (
            feasible_projection_decode(codebook, observation, whitening).target_answer == truth
        )
        correct["identity"] += (
            feasible_projection_decode(codebook, observation, np.eye(dimension)).target_answer
            == truth
        )
    assert correct["full"] / trials > 0.95
    assert correct["full"] / trials > correct["identity"] / trials + 0.2


def test_decode_validates_inputs(figure3_instance):
    codebook = build_codebook(figure3_instance, (figure3_instance.target,))
    with pytest.raises(ValueError):
        feasible_projection_decode(codebook, np.zeros(4), np.eye(5))
