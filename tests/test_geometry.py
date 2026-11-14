import numpy as np
import pytest
from conftest import make_covariance_model

from qcv.codebook import build_codebook
from qcv.geometry import (
    normalized_margin,
    stack_observation,
    target_class_separation,
    whitened_residual_radius,
    whitening_for_mode,
)


def _brute_force_separation(codebook, whitening):
    best = np.inf
    answers = list(codebook.target_classes)
    for i, left in enumerate(answers):
        for right in answers[i + 1 :]:
            for a in codebook.class_codewords(left):
                for b in codebook.class_codewords(right):
                    best = min(best, np.linalg.norm(whitening @ (a - b)))
    return best


def test_separation_hand_computed_under_identity(figure3_instance):
    compiled = figure3_instance
    target, bicycles, motorcycles, _ = compiled.all_questions
    identity = np.eye
    # Target alone: two one-hot blocks differ in two coordinates.
    codebook = build_codebook(compiled, (target,))
    assert target_class_separation(codebook, identity(5)) == pytest.approx(np.sqrt(2.0))
    # {d, b}: (d=0, b=1) and (d=1, b=1) are both feasible, so b adds nothing to the minimum.
    codebook = build_codebook(compiled, (target, bicycles))
    assert target_class_separation(codebook, identity(8)) == pytest.approx(np.sqrt(2.0))
    # {d, b, m}: a different d forces a different b or m, i.e. two differing blocks.
    codebook = build_codebook(compiled, (target, bicycles, motorcycles))
    assert target_class_separation(codebook, identity(11)) == pytest.approx(2.0)


def test_separation_matches_brute_force_for_random_metric(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    generator = np.random.default_rng(0)
    mixing = generator.normal(size=(codebook.dimension, codebook.dimension))
    whitening = mixing @ mixing.T + np.eye(codebook.dimension)
    assert target_class_separation(codebook, whitening) == pytest.approx(
        _brute_force_separation(codebook, whitening)
    )


def test_separation_scales_inversely_with_noise_scale(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    identity = np.eye(codebook.dimension)
    base = target_class_separation(codebook, identity)
    assert target_class_separation(codebook, 0.5 * identity) == pytest.approx(0.5 * base)


def test_separation_is_zero_for_a_single_target_class(compile_question):
    compiled = compile_question("clevr", "Is cube a left of cube b?")
    codebook = build_codebook(compiled, (compiled.target,))
    assert len(codebook.target_classes) == 2
    one_class = type(codebook)(
        structure_key="k",
        questions=codebook.questions,
        signature=codebook.signature,
        answer_indices=codebook.answer_indices[:1],
        codewords=codebook.codewords[:1],
        target_classes={"yes": np.array([0])},
        block_slices=codebook.block_slices,
    )
    assert target_class_separation(one_class, np.eye(2)) == 0.0


def test_whitening_modes(figure3_instance):
    codebook = build_codebook(figure3_instance, (figure3_instance.target,))
    covariance = np.diag([1.0, 4.0, 9.0, 16.0, 25.0]) + 0.5
    model = make_covariance_model(codebook, covariance)
    assert whitening_for_mode(model, "full") is not None
    full = whitening_for_mode(model, "full")
    assert np.allclose(full @ covariance @ full, np.eye(5))
    diagonal = whitening_for_mode(model, "diagonal")
    assert np.allclose(diagonal, np.diag(1.0 / np.sqrt(np.diag(covariance))))
    assert np.array_equal(whitening_for_mode(model, "identity"), np.eye(5))
    with pytest.raises(ValueError):
        whitening_for_mode(model, "block")


def test_residual_radius_and_normalised_margin():
    whitening = np.diag([2.0, 1.0])
    radius = whitened_residual_radius(np.array([0.5, 1.0]), np.array([0.0, 1.0]), whitening)
    assert radius == pytest.approx(1.0)
    assert normalized_margin(radius, 4.0) == pytest.approx(0.5)
    assert normalized_margin(radius, 0.0) == np.inf


def test_stack_observation_orders_blocks_and_validates(figure3_instance):
    target, bicycles = figure3_instance.all_questions[:2]
    blocks = {"q0": np.full(5, 0.2), "q0001": np.array([0.7, 0.2, 0.1])}
    stacked = stack_observation((target, bicycles), blocks)
    assert np.allclose(stacked, [0.2] * 5 + [0.7, 0.2, 0.1])
    with pytest.raises(KeyError):
        stack_observation((target, bicycles), {"q0": blocks["q0"]})
    with pytest.raises(ValueError):
        stack_observation((target,), {"q0": np.full(4, 0.25)})
    with pytest.raises(ValueError):
        stack_observation((target,), {"q0": np.full(5, 0.3)})
