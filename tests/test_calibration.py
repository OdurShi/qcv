import numpy as np
import pytest
from conftest import make_record

from qcv.calibration import fit_calibration, probabilities_for_record, question_subsets
from qcv.codebook import build_codebook
from qcv.types import CalibrationArtifacts


def _scenes():
    return [
        {"d": str(b - m), "b": str(b), "m": str(m), "t": str(b + m)}
        for b in range(3)
        for m in range(3)
    ]


def _calibration_records(figure3_instance, count=120, seed=0):
    """Records whose believed answers are wrong for a random subset of the questions."""
    generator = np.random.default_rng(seed)
    scenes = _scenes()
    records = []
    for index in range(count):
        gold = scenes[index % len(scenes)]
        believed = dict(gold)
        for question in figure3_instance.all_questions:
            if generator.random() < 0.3:
                believed[question.variable] = str(generator.choice(question.role.alphabet))
        strength = {
            q.variable: float(generator.uniform(1.0, 4.0)) for q in figure3_instance.all_questions
        }
        records.append(make_record(figure3_instance, believed, strength, gold, gold["d"]))
    return records


def test_question_subsets_enumerate_in_compiler_order(figure3_instance):
    subsets = [tuple(q.variable for q in s) for s in question_subsets(figure3_instance, 2)]
    assert subsets == [
        ("d",),
        ("d", "b"),
        ("d", "m"),
        ("d", "t"),
        ("d", "b", "m"),
        ("d", "b", "t"),
        ("d", "m", "t"),
    ]
    assert len(list(question_subsets(figure3_instance, 9))) == 8
    assert len(list(question_subsets(figure3_instance, 2, min_auxiliaries=2))) == 3


def test_fit_calibration_produces_one_model_per_role_and_signature(figure3_instance):
    records = _calibration_records(figure3_instance)
    artifacts, summary = fit_calibration(records, max_budget=4)
    assert summary.records_seen == 120
    assert summary.role_models == 4
    assert summary.covariance_models == 8  # all subsets of three auxiliaries
    assert summary.unsupported_signatures == 0
    assert summary.skipped_incomplete_assignments == 0
    for question in figure3_instance.all_questions:
        assert artifacts.temperature_for(question.role) > 0
        assert 0.0 <= artifacts.role_reliability(question.role) <= 1.0
    for questions in question_subsets(figure3_instance, 3):
        codebook = build_codebook(figure3_instance, questions)
        model = artifacts.covariance_for(codebook.signature)
        assert model.dimension == codebook.dimension
        assert model.sample_count == 120
        assert model.is_positive_definite
        assert np.allclose(
            model.whitening @ model.covariance @ model.whitening, np.eye(model.dimension), atol=1e-8
        )


def test_max_budget_limits_the_fitted_signatures(figure3_instance):
    artifacts, _ = fit_calibration(_calibration_records(figure3_instance), max_budget=2)
    assert len(artifacts.covariances) == 4  # target alone and the three single-auxiliary sets


def test_covariance_is_fitted_on_observation_minus_true_codeword(figure3_instance):
    records = _calibration_records(figure3_instance, count=60)
    artifacts, _ = fit_calibration(records, max_budget=2)
    target, bicycles = figure3_instance.all_questions[:2]
    codebook = build_codebook(figure3_instance, (target, bicycles))
    residuals = []
    for record in records:
        probabilities = probabilities_for_record(record, artifacts)
        observation = np.concatenate([probabilities["q0"], probabilities["q0001"]])
        residuals.append(observation - codebook.encode_assignment(record.gold))
    residuals = np.asarray(residuals)
    centered = residuals - residuals.mean(axis=0)
    empirical = centered.T @ centered / len(records)
    model = artifacts.covariance_for(codebook.signature)
    expected = (1 - model.alpha) * empirical + model.alpha * model.mu * np.eye(8)
    assert model.mu == pytest.approx(np.trace(empirical) / 8)
    assert np.allclose(model.covariance, expected)
    assert model.alpha == pytest.approx(max(model.alpha_lw, 1 / 60))


def test_role_accuracy_is_the_argmax_accuracy(figure3_instance):
    records = _calibration_records(figure3_instance, count=90)
    artifacts, _ = fit_calibration(records, max_budget=1)
    target = figure3_instance.target
    hits = [
        target.role.alphabet[int(np.argmax(record.scores["q0"].mean_logprobs))] == record.gold["d"]
        for record in records
    ]
    assert artifacts.role_reliability(target.role) == pytest.approx(np.mean(hits))


def test_incomplete_or_infeasible_gold_does_not_contribute(figure3_instance):
    records = _calibration_records(figure3_instance, count=60)
    complete = dict(records[0].gold)
    partial = make_record(figure3_instance, complete, 3.0, {"d": complete["d"]}, complete["d"])
    infeasible_gold = {"d": "2", "b": "0", "m": "0", "t": "0"}
    infeasible = make_record(figure3_instance, infeasible_gold, 3.0, infeasible_gold, "2")
    artifacts, summary = fit_calibration([*records, partial, infeasible], max_budget=4)
    full_signature = build_codebook(figure3_instance, figure3_instance.all_questions).signature
    target_signature = build_codebook(figure3_instance, (figure3_instance.target,)).signature
    target, _, motorcycles, _ = figure3_instance.all_questions
    partly_feasible = build_codebook(figure3_instance, (target, motorcycles)).signature
    assert artifacts.covariance_for(full_signature).sample_count == 60
    assert artifacts.covariance_for(target_signature).sample_count == 62
    # (d = 2, m = 0) extends to the scene b = 2, so the infeasible record still counts there.
    assert artifacts.covariance_for(partly_feasible).sample_count == 61
    # Partial gold: all 7 sets with an auxiliary. Infeasible gold: the other 6 of those sets.
    assert summary.skipped_incomplete_assignments == 7 + 6


def test_zero_residual_variance_is_outside_coverage(figure3_instance):
    scenes = _scenes()
    records = [make_record(figure3_instance, gold, 60.0, gold, gold["d"]) for gold in scenes * 3]
    artifacts, summary = fit_calibration(records, max_budget=4)
    assert artifacts.covariances == {}
    assert summary.unsupported_signatures == 8
    assert all("zero" in reason for reason in artifacts.metadata["covariance_failures"].values())


def test_artifacts_round_trip(figure3_instance, tmp_path):
    artifacts, _ = fit_calibration(_calibration_records(figure3_instance), max_budget=3)
    artifacts.save(tmp_path / "artifacts")
    loaded = CalibrationArtifacts.load(tmp_path / "artifacts")
    assert loaded.temperatures == artifacts.temperatures
    assert loaded.role_accuracy == artifacts.role_accuracy
    assert set(loaded.covariances) == set(artifacts.covariances)
    for signature, model in artifacts.covariances.items():
        other = loaded.covariances[signature]
        assert np.array_equal(other.covariance, model.covariance)
        assert np.array_equal(other.whitening, model.whitening)
        assert (other.alpha, other.mu, other.sample_count) == (
            model.alpha,
            model.mu,
            model.sample_count,
        )
        assert other.fingerprint == model.fingerprint


def test_fit_calibration_validates_inputs(figure3_instance):
    with pytest.raises(ValueError):
        fit_calibration([], max_budget=3)
    with pytest.raises(ValueError):
        fit_calibration(_calibration_records(figure3_instance, count=9), max_budget=0)
