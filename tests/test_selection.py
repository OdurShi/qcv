import math

import pytest
from conftest import ALL_EXAMPLES, identity_artifacts, random_artifacts

from qcv.codebook import build_codebook
from qcv.selection import (
    TIE_TOLERANCE,
    _improves,
    exact_select,
    fixed_select,
    greedy_select,
    random_select,
    reliability_select,
    set_separation,
)


def _ids(result):
    return [question.question_id for question in result.selected]


def test_ties_are_broken_by_template_index(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    # Under the identity metric every single auxiliary leaves d_tar at sqrt(2): a three-way tie.
    for question in figure3_instance.auxiliaries:
        value = set_separation(figure3_instance, (figure3_instance.target, question), artifacts)
        assert value == pytest.approx(math.sqrt(2.0))
    result = greedy_select(figure3_instance, 2, artifacts, "identity")
    assert _ids(result) == ["q0001"]
    assert result.separation == pytest.approx(math.sqrt(2.0))
    assert result.evaluated_subsets == 3


def test_greedy_and_exact_agree_on_the_figure3_pool(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    greedy = greedy_select(figure3_instance, 3, artifacts, "identity")
    exact = exact_select(figure3_instance, 3, artifacts, "identity")
    # Any two auxiliaries pin down d, so d_tar = 2 for all pairs; ties go to (b, m).
    assert _ids(greedy) == _ids(exact) == ["q0001", "q0002"]
    assert greedy.separation == exact.separation == pytest.approx(2.0)
    assert exact.evaluated_subsets == 3


@pytest.mark.parametrize(("task", "question"), ALL_EXAMPLES)
def test_greedy_equals_exact_where_it_must(compile_question, task, question):
    compiled = compile_question(task, question)
    artifacts = random_artifacts(compiled, seed=1)
    for metric in ("full", "diagonal", "identity"):
        # One greedy step is an exhaustive search over single auxiliaries.
        greedy = greedy_select(compiled, 2, artifacts, metric)
        exact = exact_select(compiled, 2, artifacts, metric)
        assert _ids(greedy) == _ids(exact)
        assert greedy.separation == pytest.approx(exact.separation)
        # With the whole pool selected there is a single candidate set.
        full_budget = len(compiled.auxiliaries) + 1
        greedy = greedy_select(compiled, full_budget, artifacts, metric)
        exact = exact_select(compiled, full_budget, artifacts, metric)
        assert _ids(greedy) == _ids(exact) == [q.question_id for q in compiled.auxiliaries]
        assert greedy.separation == pytest.approx(exact.separation)
        # In between, exhaustive search can only match or beat greedy on the objective.
        greedy = greedy_select(compiled, 3, artifacts, metric)
        exact = exact_select(compiled, 3, artifacts, metric)
        assert exact.separation >= greedy.separation - 1e-9
        assert len(greedy.selected) == len(exact.selected) == 2


def test_selection_is_deterministic_and_in_compiler_order(compile_question):
    compiled = compile_question("synthetic", "How many more red cubes than blue spheres are there?")
    artifacts = random_artifacts(compiled, seed=3)
    first = greedy_select(compiled, 5, artifacts)
    second = greedy_select(compiled, 5, artifacts)
    assert _ids(first) == _ids(second)
    assert _ids(first) == sorted(_ids(first))
    assert len(first.selected) == 4


def test_exact_falls_back_to_greedy_above_the_subset_limit(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    limited = exact_select(figure3_instance, 3, artifacts, "identity", subset_limit=1)
    assert _ids(limited) == _ids(greedy_select(figure3_instance, 3, artifacts, "identity"))


def test_greedy_skips_unsupported_signatures(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    target, bicycles = figure3_instance.all_questions[:2]
    del artifacts.covariances[build_codebook(figure3_instance, (target, bicycles)).signature]
    assert _ids(greedy_select(figure3_instance, 2, artifacts, "identity")) == ["q0002"]
    with pytest.raises(KeyError):
        set_separation(figure3_instance, (target, bicycles), artifacts)


def test_improvement_requires_more_than_numerical_noise():
    assert _improves(1.0, -math.inf)
    assert not _improves(1.0 + 0.5 * TIE_TOLERANCE, 1.0)
    assert _improves(1.0 + 1e-9, 1.0)


def test_fixed_probes_take_the_first_templates(figure3_instance):
    assert _ids(fixed_select(figure3_instance, 3)) == ["q0001", "q0002"]
    assert _ids(fixed_select(figure3_instance, 1)) == []


def test_reliability_selects_the_most_accurate_roles(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    bicycles, motorcycles, total = figure3_instance.auxiliaries
    artifacts.role_accuracy.update(
        {bicycles.role.key: 0.6, motorcycles.role.key: 0.9, total.role.key: 0.8}
    )
    assert _ids(reliability_select(figure3_instance, 3, artifacts)) == ["q0002", "q0003"]
    # Equal reliabilities fall back to compiler order.
    artifacts.role_accuracy.update(dict.fromkeys(artifacts.role_accuracy, 0.5))
    assert _ids(reliability_select(figure3_instance, 2, artifacts)) == ["q0001"]


def test_random_selection_is_seeded_per_instance(compile_question):
    question = "How many more red cubes than blue spheres are there?"
    first = compile_question("synthetic", question, "instance-1")
    second = compile_question("synthetic", question, "instance-2")
    assert _ids(random_select(first, 5, seed=0)) == _ids(random_select(first, 5, seed=0))
    draws = {tuple(_ids(random_select(first, 5, seed=seed))) for seed in range(20)}
    assert len(draws) > 1
    by_instance = {tuple(_ids(random_select(c, 5, seed=0))) for c in (first, second)}
    assert all(len(ids) == 4 and list(ids) == sorted(ids) for ids in draws | by_instance)


def test_budget_validation(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    with pytest.raises(KeyError):
        greedy_select(figure3_instance, 5, artifacts)
    with pytest.raises(KeyError):
        fixed_select(figure3_instance, 5)
    with pytest.raises(ValueError):
        greedy_select(figure3_instance, 0, artifacts)
