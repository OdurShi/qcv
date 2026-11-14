import numpy as np
import pytest

from qcv.constraints import evaluate_constraint
from qcv.synthetic import SyntheticConfig, SyntheticScorer, generate_raw_records
from qcv.types import CompiledInstance


def _compiled(registry, record) -> CompiledInstance:
    result = registry.compiler_for("synthetic").compile(record["instance_id"], record["question"])
    assert result.compiled is not None
    return result.compiled


def test_records_are_deterministic_and_cover_both_rules(registry):
    config = SyntheticConfig(seed=3)
    first = generate_raw_records(200, "test", config)
    assert first == generate_raw_records(200, "test", config)
    assert first != generate_raw_records(200, "test", SyntheticConfig(seed=4))
    assert (
        first[0]["instance_id"] != generate_raw_records(1, "calibration", config)[0]["instance_id"]
    )
    rules = {record["metadata"]["rule"] for record in first}
    assert rules == {"count_difference", "spatial_inverse"}


def test_scene_states_satisfy_every_constraint(registry):
    for record in generate_raw_records(150, "test", SyntheticConfig(seed=0)):
        compiled = _compiled(registry, record)
        assert set(record["gold"]) == {q.variable for q in compiled.all_questions}
        assert record["answer"] == record["gold"][compiled.target.variable]
        assert all(evaluate_constraint(c, record["gold"]) for c in compiled.constraints)


def test_scores_are_normalised_log_distributions(registry):
    config = SyntheticConfig(seed=0)
    scorer = SyntheticScorer(config)
    record = generate_raw_records(1, "test", config)[0]
    compiled = _compiled(registry, record)
    for question in compiled.all_questions:
        scores = scorer.score_question(question, compiled.target, record["gold"], "id")
        assert len(scores) == len(question.role.alphabet)
        assert np.sum(np.exp(scores)) == pytest.approx(1.0)
        assert scores == scorer.score_question(question, compiled.target, record["gold"], "id")


@pytest.mark.parametrize("correlation", [0.0, 0.8])
def test_error_rate_is_marginal_and_correlation_only_couples_restatements(registry, correlation):
    config = SyntheticConfig(
        error_rate=0.25, correlation=correlation, relation_fraction=0.0, seed=1
    )
    scorer = SyntheticScorer(config)
    errors = {"difference_ab": [], "restated_ab": [], "difference_ba": []}
    for record in generate_raw_records(3000, "test", config):
        compiled = _compiled(registry, record)
        for question in compiled.all_questions:
            if question.variable in errors:
                percept = scorer.percept(
                    question, compiled.target, record["gold"], compiled.instance_id
                )
                truth = question.role.answer_index(record["gold"][question.variable])
                errors[question.variable].append(percept != truth)
    target, restated, inverse = (np.array(errors[name], dtype=float) for name in errors)
    for column in (target, restated, inverse):
        assert column.mean() == pytest.approx(0.25, abs=0.03)
    assert np.corrcoef(target, restated)[0, 1] == pytest.approx(correlation, abs=0.06)
    assert abs(np.corrcoef(target, inverse)[0, 1]) < 0.06


def test_config_validation():
    with pytest.raises(ValueError):
        SyntheticConfig(correlation=1.5)
    with pytest.raises(ValueError):
        SyntheticConfig(error_rate=-0.1)
    with pytest.raises(ValueError):
        SyntheticConfig(noise=-1.0)
