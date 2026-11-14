import dataclasses

import numpy as np
import pytest
from conftest import identity_artifacts, make_record

from qcv.codebook import build_codebook
from qcv.inference import InferenceEngine, InferenceOptions, coverage_reason, infer_record

TRUE_SCENE = {"d": "2", "b": "2", "m": "0", "t": "2"}


def _wrong_target_record(figure3_instance, **kwargs):
    """The target block prefers d = 1 although the scene has d = 2; auxiliaries are right."""
    believed = dict(TRUE_SCENE, d="1")
    strength = {"d": 1.0, "b": 4.0, "m": 4.0, "t": 4.0}
    return make_record(figure3_instance, believed, strength, **kwargs)


@pytest.mark.parametrize("selection", ["qcv", "exact", "fixed", "random", "reliability"])
def test_covered_record_is_corrected_by_every_selector(figure3_instance, selection):
    artifacts = identity_artifacts(figure3_instance)
    options = InferenceOptions(budget=3, selection=selection, metric="identity")
    prediction = infer_record(_wrong_target_record(figure3_instance), artifacts, options)
    assert prediction.covered and prediction.reason == "qcv"
    assert prediction.target_only_answer == "1"
    assert prediction.final_answer == "2"
    assert len(prediction.selected_question_ids) == 2
    assert [question_id for question_id, _ in prediction.measurements] == [
        "q0",
        *prediction.selected_question_ids,
    ]
    assert prediction.measurements[0] == ("q0", "1")
    assert prediction.target_class_separation == pytest.approx(2.0)
    assert prediction.target_margin > 0 and prediction.projection_gap >= 0
    assert (prediction.budget, prediction.selection, prediction.metric) == (
        3,
        selection,
        "identity",
    )


def test_budget_one_under_identity_metric_is_target_only(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    options = InferenceOptions(budget=1, metric="identity")
    prediction = infer_record(_wrong_target_record(figure3_instance), artifacts, options)
    assert prediction.covered
    assert prediction.selected_question_ids == ()
    assert prediction.final_answer == prediction.target_only_answer == "1"


def test_fallback_when_the_pool_is_too_small(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    prediction = infer_record(
        _wrong_target_record(figure3_instance), artifacts, InferenceOptions(budget=5)
    )
    assert not prediction.covered
    assert prediction.reason == "insufficient_valid_auxiliaries"
    assert prediction.final_answer == prediction.target_only_answer == "1"
    assert prediction.selected_question_ids == ()
    assert prediction.measurements == (("q0", "1"),)
    assert prediction.target_margin is None and prediction.target_class_separation is None


def test_fallback_when_a_temperature_is_missing(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    del artifacts.temperatures[figure3_instance.auxiliaries[2].role.key]
    prediction = infer_record(
        _wrong_target_record(figure3_instance), artifacts, InferenceOptions(budget=2)
    )
    assert not prediction.covered
    assert prediction.reason.startswith("missing_temperature")
    assert prediction.final_answer == "1"


def test_required_signatures_depend_on_the_selection_method(figure3_instance):
    record = _wrong_target_record(figure3_instance)
    artifacts = identity_artifacts(figure3_instance)
    target, _, motorcycles, total = figure3_instance.all_questions
    missing = build_codebook(figure3_instance, (target, motorcycles, total)).signature
    del artifacts.covariances[missing]

    # Greedy and exhaustive selection need every candidate set with 1..B-1 auxiliaries.
    for selection in ("qcv", "exact"):
        options = InferenceOptions(budget=3, selection=selection, metric="identity")
        prediction = infer_record(record, artifacts, options)
        assert not prediction.covered
        assert prediction.reason == f"unsupported_signature:{missing}"
        assert prediction.final_answer == "1"
    # Fixed probes only need the signature of the set they actually use, here {d, b, m}.
    options = InferenceOptions(budget=3, selection="fixed", metric="identity")
    prediction = infer_record(record, artifacts, options)
    assert prediction.covered and prediction.final_answer == "2"
    assert coverage_reason(record, artifacts, options, figure3_instance.auxiliaries[:2]) is None
    assert coverage_reason(record, artifacts, options, figure3_instance.auxiliaries[1:]) is not None
    # Non-strict coverage lets greedy skip the unsupported candidate set.
    options = InferenceOptions(budget=3, selection="qcv", metric="identity", strict_coverage=False)
    assert infer_record(record, artifacts, options).covered


def test_non_positive_definite_covariance_is_outside_coverage(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    target, bicycles = figure3_instance.all_questions[:2]
    signature = build_codebook(figure3_instance, (target, bicycles)).signature
    model = artifacts.covariances[signature]
    singular = np.zeros_like(model.covariance)
    artifacts.covariances[signature] = dataclasses.replace(model, covariance=singular)
    prediction = infer_record(
        _wrong_target_record(figure3_instance), artifacts, InferenceOptions(budget=2)
    )
    assert not prediction.covered
    assert prediction.reason == f"non_positive_definite:{signature}"


def test_inference_never_reads_annotations(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    options = InferenceOptions(budget=3, metric="identity")
    plain = infer_record(_wrong_target_record(figure3_instance), artifacts, options)
    annotated = infer_record(
        _wrong_target_record(figure3_instance, gold=TRUE_SCENE, target_answer="2"),
        artifacts,
        options,
    )
    misleading = infer_record(
        _wrong_target_record(
            figure3_instance, gold={"d": "0", "b": "0", "m": "0", "t": "0"}, target_answer="0"
        ),
        artifacts,
        options,
    )
    assert plain.to_dict() == annotated.to_dict() == misleading.to_dict()


def test_engine_matches_single_record_inference_and_is_deterministic(compile_question):
    question = "How many more red cubes than blue spheres are there?"
    scene = {
        "difference_ab": "1", "count_a": "3", "count_b": "2", "restated_ab": "1",
        "difference_ba": "-1", "total_ab": "5", "recomputed_ab": "1",
        "comparison_ab": "greater", "more_a": "yes",
    }  # fmt: skip
    records = []
    for index in range(6):
        compiled = compile_question("synthetic", question, f"instance-{index}")
        believed = dict(scene, difference_ab=str(index - 2))
        records.append(make_record(compiled, believed, 2.0))
    artifacts = identity_artifacts(records[0].compiled, max_auxiliaries=2)
    options = InferenceOptions(budget=3, metric="identity")
    engine = InferenceEngine(artifacts, options)
    batch = [engine.infer(record).to_dict() for record in records]
    assert batch == [infer_record(record, artifacts, options).to_dict() for record in records]
    assert batch == [engine.infer(record).to_dict() for record in records]
    assert {row["final_answer"] for row in batch} == {"1"}


def test_options_validation():
    with pytest.raises(ValueError):
        InferenceOptions(budget=0)
    with pytest.raises(ValueError):
        InferenceOptions(selection="oracle")
    with pytest.raises(ValueError):
        InferenceOptions(metric="block")
