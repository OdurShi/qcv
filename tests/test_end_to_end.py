"""End-to-end runs of the real command line interface on synthetic data."""

from __future__ import annotations

import json

import pytest

from qcv.cli import main
from qcv.codebook import build_codebook
from qcv.pipeline import load_predictions, load_score_records
from qcv.utils import read_jsonl


def _run(*arguments) -> None:
    main([str(argument) for argument in arguments])


def test_cli_pipeline_from_raw_records_to_comparison(tmp_path, capsys):
    """synth -> compile -> score -> calibrate -> infer -> baseline -> evaluate -> compare."""
    simulator = ("--error-rate", "0.1", "--correlation", "0.8", "--seed", "5")
    _run("synth", "--output-dir", tmp_path, "--calibration-size", "300", "--test-size", "200",
         *simulator)  # fmt: skip
    for split in ("calibration", "test"):
        _run("compile", "--input", tmp_path / f"{split}_raw.jsonl",
             "--output", tmp_path / f"{split}_compiled.jsonl")  # fmt: skip
        _run("score", "--backend", "synthetic", "--input", tmp_path / f"{split}_compiled.jsonl",
             "--output", tmp_path / f"{split}_scores.jsonl", *simulator)  # fmt: skip
    _run("calibrate", "--input", tmp_path / "calibration_scores.jsonl",
         "--output-dir", tmp_path / "artifacts", "--max-budget", "3")  # fmt: skip
    scores, artifacts = tmp_path / "test_scores.jsonl", tmp_path / "artifacts"
    _run("infer", "--input", scores, "--artifacts", artifacts,
         "--output", tmp_path / "qcv.jsonl", "--budget", "3")  # fmt: skip
    _run("baseline", "--input", scores, "--artifacts", artifacts,
         "--output", tmp_path / "target.jsonl", "--method", "target-only")  # fmt: skip
    _run("evaluate", "--scores", scores, "--predictions", tmp_path / "qcv.jsonl",
         "--artifacts", artifacts, "--output", tmp_path / "qcv.json")  # fmt: skip
    _run("compare", "--scores", scores, "--first", tmp_path / "qcv.jsonl",
         "--second", tmp_path / "target.jsonl", "--output", tmp_path / "compare.json",
         "--bootstrap-replicates", "200")  # fmt: skip
    capsys.readouterr()

    compiled = list(read_jsonl(tmp_path / "test_compiled.jsonl"))
    assert len(compiled) == 200 and all(row["compile_status"] == "compiled" for row in compiled)
    records = load_score_records(scores)
    assert all(set(r.scores) == {q.question_id for q in r.compiled.all_questions} for r in records)
    assert {p.name for p in artifacts.iterdir()} >= {
        "temperatures.json", "role_accuracy.json", "covariances.npz",
        "covariance_index.json", "metadata.json", "calibration_summary.json",
    }  # fmt: skip
    metrics = json.loads((tmp_path / "qcv.json").read_text(encoding="utf-8"))
    comparison = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))
    assert metrics["total"] == 200 and metrics["coverage"] == 1.0
    assert comparison["paired_examples"] == 200
    assert comparison["accuracy_difference"] == pytest.approx(
        metrics["overall_accuracy"] - metrics["target_only_accuracy"]
    )
    low, high = comparison["bootstrap_95_ci"]
    assert low <= comparison["accuracy_difference"] <= high
    assert 0.0 <= comparison["mcnemar_exact_p"] <= 1.0


def test_qcv_improves_on_target_only_with_margin(run_variant):
    metrics, _ = run_variant("--budget", "5")
    assert metrics["total"] == 600 and metrics["coverage"] == 1.0
    assert metrics["target_only_accuracy"] == pytest.approx(0.9, abs=0.04)
    assert metrics["overall_accuracy"] > metrics["target_only_accuracy"] + 0.03
    assert metrics["correction_rate"] > 0.5
    assert metrics["corruption_rate"] < 0.05
    assert 1.0 < metrics["effective_measurements"] <= 5.0
    assert metrics["margin_error_auroc"] > 0.7


def test_whitened_selection_and_decoding_beat_diagonal_under_correlated_errors(
    synthetic_run, run_variant
):
    """Restatements copy the target's mistakes with probability 0.8 in the simulator. The full
    residual covariance sees that redundancy; its diagonal and the identity metric cannot."""
    full, full_path = run_variant("--budget", "5")
    diagonal, _ = run_variant("--budget", "5", "--metric", "diagonal")
    identity, identity_path = run_variant("--budget", "5", "--metric", "identity")
    assert full["overall_accuracy"] > diagonal["overall_accuracy"] + 0.01
    assert full["overall_accuracy"] > identity["overall_accuracy"] + 0.01
    counting = {
        record.instance_id
        for record in load_score_records(synthetic_run["scores"])
        if record.compiled.rule_name == "count_difference"
    }
    restatements = {"q0003", "q0006"}

    def chosen(path) -> set[str]:
        return {
            question_id
            for prediction in load_predictions(path)
            if prediction.instance_id in counting
            for question_id in prediction.selected_question_ids
        }

    assert not chosen(full_path) & restatements
    assert chosen(identity_path) & restatements


@pytest.mark.parametrize(
    "arguments",
    [
        ("--budget", "5"),
        ("--budget", "3", "--metric", "diagonal"),
        ("--budget", "4", "--selection", "fixed"),
        ("--budget", "5", "--selection", "random", "--random-seed", "3"),
    ],
)
def test_structural_guarantees_hold_for_every_variant(synthetic_run, run_variant, arguments):
    metrics, path = run_variant(*arguments)
    budget = int(arguments[1])
    records = {record.instance_id: record for record in load_score_records(synthetic_run["scores"])}
    predictions = load_predictions(path)
    assert len(predictions) == len(records) == 600
    for prediction in predictions:
        record = records[prediction.instance_id]
        assert prediction.covered
        # Exactly B question evaluations are used: the target and B - 1 auxiliaries.
        assert len(prediction.selected_question_ids) == budget - 1
        assert len(prediction.measurements) == budget
        # The decoded answer belongs to a feasible codeword of the selected question set.
        questions = record.compiled.ordered_subset(prediction.selected_question_ids)
        codebook = build_codebook(record.compiled, questions)
        assert prediction.final_answer in codebook.target_classes
        assert set(prediction.energies) == set(codebook.target_classes)
        assert prediction.energies[prediction.final_answer] == min(prediction.energies.values())
        assert prediction.target_margin >= 0.0 and prediction.projection_gap >= 0.0
        assert prediction.target_class_separation > 0.0
    # Eq. (8): whenever ||W eps|| < d_tar / 2 the decoded target must be the true one.
    audit = metrics["separation_audit"]
    assert audit["examples"] == 600
    assert audit["sufficient_condition_violations"] == 0
    assert audit["sufficient_condition_rate"] > 0.05
    assert audit["bins"]["kappa<0.8"]["final_accuracy"] == 1.0


def test_budget_beyond_the_pool_falls_back_to_target_only(synthetic_run, tmp_path, capsys):
    scores, artifacts = synthetic_run["scores"], tmp_path / "artifacts"
    _run("calibrate", "--input", synthetic_run["calibration_scores"],
         "--output-dir", artifacts, "--max-budget", "2")  # fmt: skip
    # B = 9 needs eight auxiliaries (only the counting rule has them) and their covariances
    # (this calibration stopped at B = 2), so every example uses the fallback.
    _run("infer", "--input", scores, "--artifacts", artifacts,
         "--output", tmp_path / "b9.jsonl", "--budget", "9")  # fmt: skip
    _run("evaluate", "--scores", scores, "--predictions", tmp_path / "b9.jsonl",
         "--artifacts", artifacts, "--output", tmp_path / "b9.json")  # fmt: skip
    capsys.readouterr()
    metrics = json.loads((tmp_path / "b9.json").read_text(encoding="utf-8"))
    assert metrics["coverage"] == 0.0 and metrics["covered_accuracy"] is None
    assert metrics["overall_accuracy"] == metrics["target_only_accuracy"]
    assert metrics["correction_rate"] == 0.0 and metrics["corruption_rate"] == 0.0
    predictions = load_predictions(tmp_path / "b9.jsonl")
    reasons = {prediction.reason.split(":")[0] for prediction in predictions}
    assert reasons == {"insufficient_valid_auxiliaries", "unsupported_signature"}
    assert all(p.final_answer == p.target_only_answer and not p.covered for p in predictions)


def test_inference_is_reproducible(synthetic_run, run_variant, tmp_path, capsys):
    _, reference = run_variant("--budget", "5")
    _run("infer", "--input", synthetic_run["scores"], "--artifacts", synthetic_run["artifacts"],
         "--output", tmp_path / "again.jsonl", "--budget", "5")  # fmt: skip
    capsys.readouterr()
    assert reference.read_bytes() == (tmp_path / "again.jsonl").read_bytes()


def test_repeated_target_baseline_is_redundant(synthetic_run, run_variant, tmp_path, capsys):
    scores, predictions = synthetic_run["scores"], tmp_path / "repeat.jsonl"
    _run("baseline", "--input", scores, "--artifacts", synthetic_run["artifacts"],
         "--output", predictions, "--method", "repeated-target", "--budget", "5")  # fmt: skip
    _run("evaluate", "--scores", scores, "--predictions", predictions,
         "--output", tmp_path / "repeat.json")  # fmt: skip
    capsys.readouterr()
    repeated = json.loads((tmp_path / "repeat.json").read_text(encoding="utf-8"))
    qcv, _ = run_variant("--budget", "5")
    assert repeated["coverage"] == 1.0 and repeated["separation_audit"] is None
    assert repeated["mean_error_correlation"] > qcv["mean_error_correlation"] + 0.1
    assert repeated["effective_measurements"] < qcv["effective_measurements"]
    assert repeated["correct_measurement_survival"] < qcv["correct_measurement_survival"]


def test_demo_command_reports_all_methods(tmp_path, capsys):
    _run("demo", "--output-dir", tmp_path / "demo", "--calibration-size", "250",
         "--test-size", "200", "--budget", "3", "--bootstrap-replicates", "100")  # fmt: skip
    report = capsys.readouterr().out
    methods = ("target-only", "repeated-target", "fixed-probes", "random-qcv", "reliability-qcv",
               "identity-qcv", "diagonal-qcv", "qcv", "exact-qcv")  # fmt: skip
    for name in methods:
        assert f"\n{name} " in report
        assert (tmp_path / "demo" / f"{name}.metrics.json").is_file()
    assert "not a result on any benchmark" in report
    assert (tmp_path / "demo" / "report.txt").read_text(encoding="utf-8") == report
    assert "wrong answers among them: 0" in report
    assert (tmp_path / "demo" / "qcv_vs_diagonal-qcv.json").is_file()
