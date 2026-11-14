"""Run a method x budget sweep described by a YAML config on one score file.

For every budget the script runs the configured baselines and QCV variants,
evaluates each prediction file, compares each method with the reference method
(paired bootstrap interval, exact McNemar test, Holm-Bonferroni adjustment within
the budget) and writes

  <method>.B<budget>.predictions.jsonl / .metrics.json
  sweep.csv           one row of scalar metrics per (method, budget)
  comparisons.json    paired comparisons against the reference method
  correlations.json   Spearman correlation of the correction rate with
                      rho_bar_B, S_B and B_eff across the sweep cells
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import spearmanr

from qcv.comparison import compare_prediction_files
from qcv.inference import InferenceOptions
from qcv.metrics import holm_bonferroni
from qcv.pipeline import baseline_jsonl, evaluate_jsonl, infer_jsonl
from qcv.utils import json_dump

GEOMETRY_KEYS = (
    "mean_error_correlation",
    "correct_measurement_survival",
    "effective_measurements",
)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("budgets", "methods"):
        if key not in config:
            raise ValueError(f"Sweep config {path} lacks the required key {key!r}")
    return config


def spearman_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Spearman r_s between correction rate and each geometry statistic over sweep cells."""
    summary: dict[str, Any] = {}
    for key in GEOMETRY_KEYS:
        pairs = [
            (row[key], row["correction_rate"])
            for row in rows
            if row.get(key) is not None and row.get("correction_rate") is not None
        ]
        if len(pairs) < 3:
            summary[key] = {"cells": len(pairs), "spearman_r": None}
            continue
        statistic = spearmanr([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        summary[key] = {"cells": len(pairs), "spearman_r": float(statistic.statistic)}
    return summary


def run_sweep(
    config: dict[str, Any], scores: Path, artifacts: Path, output_dir: Path
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("random_seed", 0))
    replicates = int(config.get("bootstrap_replicates", 10_000))
    reference = config.get("reference")
    rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}

    for budget in config["budgets"]:
        predictions: dict[str, Path] = {}
        for name in config.get("baselines", []):
            predictions[name] = output_dir / f"{name}.B{budget}.predictions.jsonl"
            baseline_jsonl(scores, artifacts, predictions[name], name, budget=budget, seed=seed)
        for name, spec in config["methods"].items():
            predictions[name] = output_dir / f"{name}.B{budget}.predictions.jsonl"
            options = InferenceOptions(
                budget=budget,
                selection=spec["selection"],
                metric=spec.get("metric", "full"),
                random_seed=seed,
            )
            infer_jsonl(scores, artifacts, predictions[name], options)
        for name, path in predictions.items():
            metrics = evaluate_jsonl(
                scores, path, output_dir / f"{name}.B{budget}.metrics.json", artifacts
            )
            scalars = {key: value for key, value in metrics.items() if not isinstance(value, dict)}
            rows.append({"method": name, "budget": budget, **scalars})
        if reference in predictions:
            results = {
                name: compare_prediction_files(
                    scores,
                    predictions[reference],
                    path,
                    output_dir / f"{reference}_vs_{name}.B{budget}.json",
                    bootstrap_replicates=replicates,
                    seed=seed,
                )
                for name, path in predictions.items()
                if name != reference
            }
            adjusted = holm_bonferroni(
                {name: result["mcnemar_exact_p"] for name, result in results.items()}
            )
            for name, result in results.items():
                result["mcnemar_holm_p"] = adjusted[name]
            comparisons[f"B{budget}"] = results

    columns = [
        "method",
        "budget",
        *sorted({key for row in rows for key in row} - {"method", "budget"}),
    ]
    with (output_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    json_dump(output_dir / "comparisons.json", comparisons)
    qcv_cells = [row for row in rows if row["method"] in config["methods"]]
    json_dump(output_dir / "correlations.json", spearman_summary(qcv_cells))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/budget_sweep.yaml")
    parser.add_argument("--scores", required=True, help="test score file (JSONL)")
    parser.add_argument("--artifacts", required=True, help="calibration artifact directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rows = run_sweep(
        load_config(Path(args.config)),
        Path(args.scores),
        Path(args.artifacts),
        Path(args.output_dir),
    )
    for row in rows:
        coverage = row["coverage"]
        print(
            f"B={row['budget']} {row['method']:<16} acc={100 * row['overall_accuracy']:.1f} "
            f"cov={100 * coverage:.1f}"
        )


if __name__ == "__main__":
    main()
