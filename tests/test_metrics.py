import numpy as np
import pytest
from conftest import identity_artifacts, make_record

from qcv.inference import InferenceOptions, infer_record
from qcv.metrics import (
    answers_equal,
    auroc,
    correction_and_corruption,
    effective_measurements,
    evaluate_predictions,
    holm_bonferroni,
    mcnemar_exact,
    mean_pairwise_error_correlation,
    paired_bootstrap_accuracy_difference,
    post_failure_survival,
)
from qcv.types import Prediction


def test_answers_equal_exact_and_chartqa_relaxed():
    assert answers_equal("clevr", " Yes. ", "yes")
    assert not answers_equal("clevr", "3", "3.1")
    assert answers_equal("chartqa", "104", "100")  # within 5%
    assert not answers_equal("chartqa", "106", "100")
    assert answers_equal("chartqa", "0", "0") and not answers_equal("chartqa", "0.01", "0")
    assert answers_equal("chartqa", "greater", "Greater")


def test_auroc_hand_cases():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # Positives 0.4 and 0.9 against negatives 0.1, 0.5, 0.6: 4 of 6 pairs are ordered correctly.
    assert auroc([1, 0, 0, 0, 1], [0.4, 0.1, 0.5, 0.6, 0.9]) == pytest.approx(4 / 6)
    assert auroc([1, 1], [0.1, 0.2]) is None


def test_auroc_matches_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, size=300)
    scores = np.round(generator.normal(size=300) + labels, 1)  # rounding creates ties
    assert auroc(labels, scores) == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores))


def test_error_correlation_hand_cases():
    errors = np.array([[1, 1], [0, 0], [1, 1], [0, 0]])
    assert mean_pairwise_error_correlation(errors) == pytest.approx(1.0)
    assert mean_pairwise_error_correlation(np.array([[1, 0], [0, 1]])) == pytest.approx(-1.0)
    # Columns 0/1 identical, column 2 independent of them: mean of (1, 0, 0).
    errors = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert mean_pairwise_error_correlation(errors) == pytest.approx(1 / 3)
    # A slot that never fails has undefined correlations and is skipped.
    errors = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert mean_pairwise_error_correlation(errors) == pytest.approx(0.0)
    assert mean_pairwise_error_correlation(np.zeros((4, 3))) is None
    assert mean_pairwise_error_correlation(np.zeros((4, 1))) is None


@pytest.mark.parametrize(
    ("budget", "rho", "expected"),
    [(3, 0.68, 1.27), (9, 0.68, 1.40), (3, 0.063, 2.66), (9, 0.031, 7.21), (5, 0.0625, 4.0)],
)
def test_effective_measurements_formula(budget, rho, expected):
    """B_eff = B / (1 + (B - 1) rho); the (B, rho, B_eff) triples are quoted in the paper's text."""
    assert effective_measurements(budget, rho) == pytest.approx(expected, abs=0.01)


def test_effective_measurements_limits():
    assert effective_measurements(5, 0.0) == 5.0
    assert effective_measurements(5, 1.0) == pytest.approx(1.0)
    assert effective_measurements(5, -0.1) == 5.0  # capped at the nominal budget


def test_post_failure_survival_hand_case():
    errors = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
    # Target wrong in rows 0, 1, 3; correct others there: 2, 1, 0 of 2 -> 3 / 6.
    assert post_failure_survival(errors) == pytest.approx(0.5)
    assert post_failure_survival(np.array([[0, 1], [0, 0]])) is None
    assert post_failure_survival(np.array([[1], [0]])) is None


def test_correction_and_corruption_toy_confusion():
    initial = [True] * 6 + [False] * 4
    final = [True] * 5 + [False] + [True] * 3 + [False]
    correction, corruption = correction_and_corruption(initial, final)
    assert correction == pytest.approx(3 / 4)
    assert corruption == pytest.approx(1 / 6)
    assert correction_and_corruption([True, True], [True, False]) == (None, 0.5)


def _prediction(instance_id, covered, initial, final, margin=None, gap=None, measurements=()):
    return Prediction(
        instance_id=instance_id,
        covered=covered,
        target_only_answer=initial,
        final_answer=final,
        selected_question_ids=tuple(question_id for question_id, _ in measurements[1:]),
        target_margin=margin,
        projection_gap=gap,
        target_class_separation=None,
        reason="qcv" if covered else "fallback",
        measurements=tuple(measurements),
    )


def test_evaluate_predictions_on_a_hand_built_file(figure3_instance, compile_question):
    scene = {"d": "1", "b": "2", "m": "1", "t": "3"}

    def record(instance_id, task):
        compiled = type(figure3_instance)(
            instance_id, task, figure3_instance.target, figure3_instance.auxiliaries,
            figure3_instance.constraints, figure3_instance.rule_name,
        )  # fmt: skip
        return make_record(compiled, scene, 2.0, scene, "1")

    records = [record("a", "t1"), record("b", "t1"), record("c", "t1"), record("d", "t2")]
    right = [("q0", "1"), ("q0001", "2")]
    predictions = [
        # corrected, covered, target measurement wrong but the auxiliary right
        _prediction("a", True, "0", "1", 2.0, 0.1, [("q0", "0"), ("q0001", "2")]),
        # corrupted, covered, both measurements right
        _prediction("b", True, "1", "0", 0.1, 0.9, right),
        # unchanged and wrong, uncovered
        _prediction("c", False, "2", "2"),
        # unchanged and right, covered, auxiliary wrong
        _prediction("d", True, "1", "1", 3.0, 0.0, [("q0", "1"), ("q0001", "0")]),
    ]
    summary = evaluate_predictions(records, predictions)
    assert summary.total == 4
    assert summary.overall_accuracy == pytest.approx(0.5)
    assert summary.target_only_accuracy == pytest.approx(0.5)
    assert summary.coverage == pytest.approx(0.75)
    assert summary.covered_accuracy == pytest.approx(2 / 3)
    assert summary.uncovered_accuracy == pytest.approx(0.0)
    assert summary.correction_rate == pytest.approx(0.5)  # a corrected, c not
    assert summary.corruption_rate == pytest.approx(0.5)  # b corrupted, d kept
    # Per-task accuracies 1/3 and 1 -> macro average 2/3 (micro average is 1/2).
    assert summary.per_task["t1"]["overall_accuracy"] == pytest.approx(1 / 3)
    assert summary.per_task["t2"]["overall_accuracy"] == pytest.approx(1.0)
    assert summary.macro_overall_accuracy == pytest.approx(2 / 3)
    # Error matrix of the covered rows: [[1, 0], [0, 0], [0, 1]] -> correlation -1/2.
    assert summary.mean_error_correlation == pytest.approx(-0.5)
    assert summary.effective_measurements == pytest.approx(2.0)
    assert summary.correct_measurement_survival == pytest.approx(1.0)
    # The only wrong covered answer (b) has the smallest margin and the largest gap.
    assert summary.margin_error_auroc == 1.0
    assert summary.gap_error_auroc == 1.0
    assert summary.separation_audit is None
    assert set(summary.to_dict()) >= {"per_task", "separation_audit", "macro_overall_accuracy"}


def test_evaluate_predictions_validates_ids(figure3_instance):
    scene = {"d": "1", "b": "2", "m": "1", "t": "3"}
    record = make_record(figure3_instance, scene, 2.0, scene, "1")
    with pytest.raises(KeyError):
        evaluate_predictions([record], [_prediction("other", True, "1", "1")])
    duplicate = [_prediction("fig3", True, "1", "1")] * 2
    with pytest.raises(ValueError):
        evaluate_predictions([record], duplicate)
    unlabeled = make_record(figure3_instance, scene, 2.0)
    with pytest.raises(ValueError):
        evaluate_predictions([unlabeled], [_prediction("fig3", True, "1", "1")])


def test_separation_audit_reports_the_normalised_margin(figure3_instance):
    artifacts = identity_artifacts(figure3_instance)
    options = InferenceOptions(budget=3, metric="identity")
    scene = {"d": "2", "b": "2", "m": "0", "t": "2"}
    confident = make_record(figure3_instance, scene, 8.0, scene, "2")
    wrong_target = make_record(
        figure3_instance, dict(scene, d="1"), {"d": 3.0, "b": 0.5, "m": 0.5, "t": 0.5}, scene, "2"
    )
    wrong_target = type(wrong_target)(
        "other", wrong_target.task,
        type(figure3_instance)(
            "other", "toy", figure3_instance.target, figure3_instance.auxiliaries,
            figure3_instance.constraints, figure3_instance.rule_name,
        ),
        wrong_target.scores, wrong_target.gold, wrong_target.target_answer,
    )  # fmt: skip
    records = [confident, wrong_target]
    predictions = [infer_record(record, artifacts, options) for record in records]
    audit = evaluate_predictions(records, predictions, artifacts).separation_audit
    assert audit["examples"] == 2
    # Confident record: z is essentially the true codeword, so kappa is close to 0.
    assert audit["bins"]["kappa<0.8"] == {
        "examples": 1,
        "final_accuracy": 1.0,
        "correction_rate": None,
    }
    assert audit["bins"]["kappa>=1.2"]["examples"] == 1
    assert audit["sufficient_condition_rate"] == pytest.approx(0.5)
    assert audit["sufficient_condition_violations"] == 0


def test_paired_bootstrap_known_cases():
    same = [True, False, True, True]
    assert paired_bootstrap_accuracy_difference(same, same, replicates=200) == (0.0, 0.0, 0.0)
    estimate, low, high = paired_bootstrap_accuracy_difference([True] * 5, [False] * 5, 200)
    assert (estimate, low, high) == (1.0, 1.0, 1.0)
    generator = np.random.default_rng(0)
    first = generator.random(400) < 0.8
    second = generator.random(400) < 0.6
    estimate, low, high = paired_bootstrap_accuracy_difference(first, second, 2000, seed=1)
    assert estimate == pytest.approx(first.mean() - second.mean())
    assert low < estimate < high and low > 0.0
    assert paired_bootstrap_accuracy_difference(first, second, 2000, seed=1) == (
        estimate,
        low,
        high,
    )
    with pytest.raises(ValueError):
        paired_bootstrap_accuracy_difference([True], [True, False])


def test_mcnemar_exact_known_values():
    # 5 pairs where only the first is right, 1 where only the second is: 2 * P(X <= 1 | 6, 1/2).
    first = [True] * 5 + [False] + [True] * 4
    second = [False] * 5 + [True] + [True] * 4
    assert mcnemar_exact(first, second) == pytest.approx(2 * (1 + 6) / 64)
    assert mcnemar_exact([True, False], [True, False]) == 1.0
    assert mcnemar_exact([True, False], [False, True]) == 1.0
    assert mcnemar_exact([True] * 10, [False] * 10) == pytest.approx(2 / 2**10)


def test_holm_bonferroni_step_down():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm_bonferroni({"x": 0.9, "y": 0.8})["x"] == 1.0
