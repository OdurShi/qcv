import numpy as np
import pytest
from conftest import identity_artifacts, make_record

from qcv.baselines import baseline_prediction, repeated_target_prediction, target_only_prediction
from qcv.inference import InferenceOptions, infer_record


def _record(figure3_instance, strength=1.0):
    believed = {"d": "1", "b": "2", "m": "0", "t": "2"}
    return make_record(figure3_instance, believed, strength)


def test_target_only_is_the_argmax_of_the_target_block(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    prediction = target_only_prediction(_record(figure3_instance), artifacts)
    assert prediction.final_answer == prediction.target_only_answer == "1"
    assert prediction.covered and prediction.budget == 1
    assert prediction.measurements == (("q0", "1"),)
    fallback = infer_record(_record(figure3_instance), artifacts, InferenceOptions(budget=9))
    assert fallback.final_answer == prediction.final_answer


def test_repeated_target_is_seeded_and_uses_the_whole_budget(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    record = _record(figure3_instance)
    first = repeated_target_prediction(record, artifacts, budget=5, seed=0)
    assert first.to_dict() == repeated_target_prediction(record, artifacts, 5, seed=0).to_dict()
    assert len(first.measurements) == 5
    assert {question_id for question_id, _ in first.measurements} == {"q0"}
    assert first.target_only_answer == "1"
    draws = {
        repeated_target_prediction(record, artifacts, 5, seed=s).measurements for s in range(20)
    }
    assert len(draws) > 1
    votes = [answer for _, answer in first.measurements]
    assert votes.count(first.final_answer) == max(votes.count(answer) for answer in set(votes))


def test_repeated_target_samples_follow_the_calibrated_distribution(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    record = _record(figure3_instance, strength=np.log(6.0))  # p(d = 1) = 6 / 10
    prediction = repeated_target_prediction(record, artifacts, budget=4000, seed=1)
    share = np.mean([answer == "1" for _, answer in prediction.measurements])
    assert share == pytest.approx(0.6, abs=0.03)
    assert prediction.final_answer == "1"  # a large vote converges to the arg-max


def test_vote_ties_go_to_the_more_probable_answer(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    record = _record(figure3_instance, strength=2.0)
    for seed in range(200):
        prediction = repeated_target_prediction(record, artifacts, budget=2, seed=seed)
        answers = [answer for _, answer in prediction.measurements]
        if len(set(answers)) == 2 and "1" in answers:
            assert prediction.final_answer == "1"
            break
    else:
        pytest.fail("no tied vote involving the arg-max was drawn")


def test_baseline_dispatch(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    record = _record(figure3_instance)
    assert baseline_prediction(record, artifacts, "target-only").reason == "target_only"
    assert baseline_prediction(record, artifacts, "repeated-target", 3).reason == "repeated_target"
    with pytest.raises(ValueError):
        baseline_prediction(record, artifacts, "self-consistency")
    with pytest.raises(ValueError):
        repeated_target_prediction(record, artifacts, budget=0)
