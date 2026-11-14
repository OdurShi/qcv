import pytest

from qcv.constraints import CONSTRAINT_KINDS, evaluate_constraint
from qcv.types import ConstraintSpec

CASES = [
    ("sum_eq", ("a", "b", "t"), {}, {"a": "2", "b": "3", "t": "5"}, True),
    ("sum_eq", ("a", "b", "t"), {}, {"a": "2", "b": "3", "t": "6"}, False),
    ("difference_eq", ("a", "b", "d"), {}, {"a": "2", "b": "5", "d": "-3"}, True),
    ("difference_eq", ("a", "b", "d"), {}, {"a": "2", "b": "5", "d": "3"}, False),
    (
        "difference_eq",
        ("a", "b", "d"),
        {"tolerance": "0.5"},
        {"a": "2", "b": "1", "d": "1.4"},
        True,
    ),
    ("exists_iff_positive", ("n", "e"), {}, {"n": "0", "e": "no"}, True),
    ("exists_iff_positive", ("n", "e"), {}, {"n": "0", "e": "yes"}, False),
    ("exists_iff_positive", ("n", "e"), {}, {"n": "4", "e": "yes"}, True),
    (
        "threshold_boolean",
        ("n", "f"),
        {"operator": "gt", "threshold": 1},
        {"n": "2", "f": "yes"},
        True,
    ),
    (
        "threshold_boolean",
        ("n", "f"),
        {"operator": "gt", "threshold": 1},
        {"n": "1", "f": "yes"},
        False,
    ),
    (
        "threshold_boolean",
        ("n", "f"),
        {"operator": "eq", "threshold": 0},
        {"n": "0", "f": "yes"},
        True,
    ),
    (
        "threshold_boolean",
        ("n", "f"),
        {"operator": "le", "threshold": 0},
        {"n": "1", "f": "no"},
        True,
    ),
    (
        "inverse_relation",
        ("f", "r"),
        {"inverse_map": {"left": "right", "right": "left"}},
        {"f": "left", "r": "right"},
        True,
    ),
    (
        "inverse_relation",
        ("f", "r"),
        {"inverse_map": {"left": "right", "right": "left"}},
        {"f": "left", "r": "left"},
        False,
    ),
    ("relation_boolean", ("r", "f"), {"relation": "left"}, {"r": "left", "f": "yes"}, True),
    ("relation_boolean", ("r", "f"), {"relation": "left"}, {"r": "right", "f": "yes"}, False),
    ("comparison_from_values", ("a", "b", "c"), {}, {"a": "3", "b": "1", "c": "greater"}, True),
    ("comparison_from_values", ("a", "b", "c"), {}, {"a": "1", "b": "1", "c": "equal"}, True),
    ("comparison_from_values", ("a", "b", "c"), {}, {"a": "0", "b": "1", "c": "greater"}, False),
    ("equal", ("x", "y"), {}, {"x": "left", "y": "left"}, True),
    ("not_equal", ("x", "y"), {}, {"x": "left", "y": "left"}, False),
    ("boolean_not", ("x", "y"), {}, {"x": "yes", "y": "no"}, True),
    ("boolean_not", ("x", "y"), {}, {"x": "yes", "y": "yes"}, False),
    (
        "one_hot_relations",
        ("r", "l", "g"),
        {"labels": ["left", "right"]},
        {"r": "right", "l": "no", "g": "yes"},
        True,
    ),
    (
        "one_hot_relations",
        ("r", "l", "g"),
        {"labels": ["left", "right"]},
        {"r": "right", "l": "yes", "g": "yes"},
        False,
    ),
]


@pytest.mark.parametrize(("kind", "variables", "parameters", "assignment", "expected"), CASES)
def test_constraint_truth_table(kind, variables, parameters, assignment, expected):
    constraint = ConstraintSpec(kind=kind, variables=variables, parameters=parameters)
    assert evaluate_constraint(constraint, assignment) is expected


def test_every_constraint_kind_has_a_case():
    assert {case[0] for case in CASES} == set(CONSTRAINT_KINDS)


def test_missing_variable_and_unknown_kind_raise():
    with pytest.raises(KeyError):
        evaluate_constraint(ConstraintSpec("equal", ("x", "y")), {"x": "a"})
    with pytest.raises(ValueError):
        evaluate_constraint(ConstraintSpec("no_such_kind", ("x", "y")), {"x": "a", "y": "a"})
    with pytest.raises(ValueError):
        evaluate_constraint(ConstraintSpec("boolean_not", ("x", "y")), {"x": "maybe", "y": "no"})
