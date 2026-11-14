"""The thin entry points in scripts/ run on the synthetic fixture."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from conftest import load_script

ROOT = Path(__file__).resolve().parents[1]


def _run_script(name: str, *arguments) -> str:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *(str(a) for a in arguments)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_validate_grammars_script():
    report = json.loads(_run_script("validate_grammars.py"))
    assert report["validated"] == 17
    assert all(case["codewords"] > 0 and case["target_classes"] >= 2 for case in report["cases"])


def test_ablation_sweep_script(synthetic_run, tmp_path):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        """
budgets: [3, 5]
random_seed: 0
bootstrap_replicates: 100
reference: qcv
baselines: [target-only]
methods:
  fixed-probes: {selection: fixed, metric: full}
  diagonal-qcv: {selection: qcv, metric: diagonal}
  qcv: {selection: qcv, metric: full}
""",
        encoding="utf-8",
    )
    scores = tmp_path / "scores.jsonl"
    lines = synthetic_run["scores"].read_text(encoding="utf-8").splitlines(keepends=True)
    scores.write_text("".join(lines[:200]), encoding="utf-8")
    output = tmp_path / "sweep"
    _run_script(
        "run_ablation_sweep.py", "--config", config, "--scores", scores,
        "--artifacts", synthetic_run["artifacts"], "--output-dir", output,
    )  # fmt: skip
    with (output / "sweep.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["method"], row["budget"]) for row in rows] == [
        (method, budget)
        for budget in ("3", "5")
        for method in ("target-only", "fixed-probes", "diagonal-qcv", "qcv")
    ]
    assert all(0.0 <= float(row["overall_accuracy"]) <= 1.0 for row in rows)
    comparisons = json.loads((output / "comparisons.json").read_text(encoding="utf-8"))
    assert set(comparisons) == {"B3", "B5"}
    assert set(comparisons["B5"]) == {"target-only", "fixed-probes", "diagonal-qcv"}
    for result in comparisons["B5"].values():
        assert result["mcnemar_holm_p"] >= result["mcnemar_exact_p"]
    correlations = json.loads((output / "correlations.json").read_text(encoding="utf-8"))
    assert correlations["effective_measurements"]["cells"] == 6
    assert -1.0 <= correlations["effective_measurements"]["spearman_r"] <= 1.0


def test_shipped_sweep_configs_are_valid():
    import yaml

    from qcv.baselines import BASELINE_METHODS
    from qcv.inference import InferenceOptions

    for name in ("main_b5.yaml", "budget_sweep.yaml"):
        config = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert config["reference"] in config["methods"]
        assert set(config["baselines"]) <= set(BASELINE_METHODS)
        for budget in config["budgets"]:
            for spec in config["methods"].values():
                InferenceOptions(budget=budget, selection=spec["selection"], metric=spec["metric"])
    assert yaml.safe_load((ROOT / "configs" / "main_b5.yaml").read_text())["budgets"] == [5]
    assert yaml.safe_load((ROOT / "configs" / "budget_sweep.yaml").read_text())["budgets"] == [
        3, 5, 7, 9,
    ]  # fmt: skip


def test_make_table_macro_averages_over_inputs(tmp_path):
    header = "method,budget,overall_accuracy,coverage,covered_accuracy,correction_rate"
    for name, accuracy, coverage in (("clevr", 0.9, 1.0), ("gqa", 0.7, 0.8)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sweep.csv").write_text(
            f"{header}\nqcv,5,{accuracy},{coverage},{accuracy},0.5\nqcv,3,0.1,1.0,0.1,\n",
            encoding="utf-8",
        )
    table = _run_script(
        "make_table.py", tmp_path / "clevr" / "sweep.csv", tmp_path / "gqa" / "sweep.csv"
    )
    row = next(line for line in table.splitlines() if line.startswith("| qcv"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    # method, per-input accuracies, then macro-averaged Acc_all, Cov., Acc_cov, Corr., ...
    assert cells[:7] == ["qcv", "90.0", "70.0", "80.0", "90.0", "80.0", "50.0"]
    assert cells[7:] == ["-"] * 5
    assert "| Method | clevr | gqa | Acc_all |" in table


def test_clevr_conversion_derives_gold_from_the_scene(registry):
    convert_clevr = load_script("convert_clevr")
    scene = {
        "objects": [
            {"color": "red", "shape": "cube", "size": "large", "material": "metal"},
            {"color": "red", "shape": "sphere", "size": "small", "material": "rubber"},
            {"color": "blue", "shape": "cube", "size": "small", "material": "rubber"},
        ],
        "relationships": {"left": [[1], [], [0, 1]]},
    }
    program = [
        {"function": "scene", "inputs": [], "value_inputs": []},
        {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
        {"function": "count", "inputs": [1], "value_inputs": []},
    ]
    assert convert_clevr.execute_program(scene, program)[-1] == 2
    relate = [
        {"function": "scene", "inputs": [], "value_inputs": []},
        {"function": "filter_color", "inputs": [0], "value_inputs": ["blue"]},
        {"function": "unique", "inputs": [1], "value_inputs": []},
        {"function": "relate", "inputs": [2], "value_inputs": ["left"]},
        {"function": "exist", "inputs": [3], "value_inputs": []},
    ]
    assert convert_clevr.execute_program(scene, relate)[-1] is True

    compiler = registry.compiler_for("clevr")
    counting = compiler.compile("x", "How many red things are there?").compiled
    gold = convert_clevr.derive_gold(counting, {"answer": "2", "program": program}, scene)
    assert gold == {
        "count_a": "2",
        "exists_a": "yes",
        "count_not_a": "1",
        "count_total": "3",
        "exists_not_a": "yes",
    }
    existence = compiler.compile("x", "Are there any red things?").compiled
    exist_program = [*program[:2], {"function": "exist", "inputs": [1], "value_inputs": []}]
    gold = convert_clevr.derive_gold(existence, {"answer": "yes", "program": exist_program}, scene)
    assert gold["exists_a"] == "yes" and gold["count_a"] == "2" and gold["count_total"] == "3"
    # Every derived scene state must be a feasible codeword of the compiled pool.
    from qcv.codebook import build_codebook

    for compiled, state in ((counting, gold | {"count_a": "2"}), (existence, gold)):
        assert build_codebook(compiled, compiled.all_questions).row_of(state) is not None
