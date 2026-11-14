"""End-to-end run of the pipeline on synthetic data (no dataset, no VLM).

``run_demo`` chains exactly the stages of the command line interface:
synth -> compile -> score (synthetic backend) -> calibrate -> infer / baseline
-> evaluate -> compare, and returns a plain-text report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .comparison import compare_prediction_files
from .inference import InferenceOptions
from .pipeline import (
    baseline_jsonl,
    calibrate_jsonl,
    compile_jsonl,
    evaluate_jsonl,
    infer_jsonl,
    provide_synthetic_raw_jsonl,
    score_jsonl_synthetic,
)
from .synthetic import SyntheticConfig

METHOD_VARIANTS: dict[str, tuple[str, str]] = {
    "fixed-probes": ("fixed", "full"),
    "random-qcv": ("random", "full"),
    "reliability-qcv": ("reliability", "full"),
    "identity-qcv": ("qcv", "identity"),
    "diagonal-qcv": ("qcv", "diagonal"),
    "qcv": ("qcv", "full"),
    "exact-qcv": ("exact", "full"),
}
"""Method name -> (selection, metric). All but ``identity-qcv`` and ``exact-qcv`` are
rows of the paper's tables; those two are additional diagnostics."""

BASELINE_VARIANTS = ("target-only", "repeated-target")

_COLUMNS = (
    ("overall_accuracy", "Acc_all", 100.0),
    ("coverage", "Cov", 100.0),
    ("covered_accuracy", "Acc_cov", 100.0),
    ("correction_rate", "Corr", 100.0),
    ("corruption_rate", "Corrupt", 100.0),
    ("mean_error_correlation", "rho_B", 1.0),
    ("effective_measurements", "B_eff", 1.0),
    ("correct_measurement_survival", "S_B", 100.0),
    ("margin_error_auroc", "AUROC", 100.0),
)


def format_metrics_table(rows: dict[str, dict[str, Any]]) -> str:
    """Fixed-width table of the main metrics, one method per line."""
    header = f"{'method':<16}" + "".join(f"{title:>9}" for _, title, _ in _COLUMNS)
    lines = [header, "-" * len(header)]
    for name, metrics in rows.items():
        cells = []
        for key, _, scale in _COLUMNS:
            value = metrics.get(key)
            if value is None:
                cells.append(f"{'-':>9}")
            elif scale == 1.0:
                cells.append(f"{value:>9.2f}")
            else:
                cells.append(f"{value * scale:>9.1f}")
        lines.append(f"{name:<16}" + "".join(cells))
    return "\n".join(lines)


def run_demo(
    output_directory: str | Path,
    calibration_size: int = 1500,
    test_size: int = 2000,
    budget: int = 5,
    config: SyntheticConfig | None = None,
    bootstrap_replicates: int = 2000,
    data_dir: str | Path | None = None,
) -> str:
    """Run the synthetic pipeline and return a text report (also saved as ``report.txt``).

    The raw records are taken from ``<data_dir>/synthetic`` when the shipped
    files match the requested seed and sizes, and are generated otherwise.
    """
    config = config or SyntheticConfig()
    root = Path(output_directory)
    provide_synthetic_raw_jsonl(root, calibration_size, test_size, config, data_dir)
    for split in ("calibration", "test"):
        compile_jsonl(root / f"{split}_raw.jsonl", root / f"{split}_compiled.jsonl")
        score_jsonl_synthetic(
            root / f"{split}_compiled.jsonl", root / f"{split}_scores.jsonl", config
        )
    scores = root / "test_scores.jsonl"
    artifacts = root / "calibration_artifacts"
    calibration = calibrate_jsonl(root / "calibration_scores.jsonl", artifacts, max_budget=budget)

    rows: dict[str, dict[str, Any]] = {}
    for name in BASELINE_VARIANTS:
        predictions = root / f"{name}.predictions.jsonl"
        baseline_jsonl(scores, artifacts, predictions, name, budget=budget, seed=config.seed)
        rows[name] = evaluate_jsonl(scores, predictions, root / f"{name}.metrics.json", artifacts)
    for name, (selection, metric) in METHOD_VARIANTS.items():
        predictions = root / f"{name}.predictions.jsonl"
        options = InferenceOptions(
            budget=budget, selection=selection, metric=metric, random_seed=config.seed
        )
        infer_jsonl(scores, artifacts, predictions, options)
        rows[name] = evaluate_jsonl(scores, predictions, root / f"{name}.metrics.json", artifacts)

    comparisons = {
        other: compare_prediction_files(
            scores,
            root / "qcv.predictions.jsonl",
            root / f"{other}.predictions.jsonl",
            root / f"qcv_vs_{other}.json",
            bootstrap_replicates=bootstrap_replicates,
            seed=config.seed,
        )
        for other in ("target-only", "diagonal-qcv")
    }

    lines = [
        "QCV synthetic demo (simulated scores; not a result on any benchmark)",
        f"simulator: error_rate={config.error_rate} correlation={config.correlation} "
        f"seed={config.seed} | calibration={calibration_size} test={test_size} budget B={budget}",
        f"calibration: {calibration['role_models']} role temperatures, "
        f"{calibration['covariance_models']} covariance models, "
        f"{calibration['unsupported_signatures']} unsupported signatures",
        "",
        format_metrics_table(rows),
        "",
    ]
    for other, result in comparisons.items():
        low, high = result["bootstrap_95_ci"]
        lines.append(
            f"qcv - {other}: {100 * result['accuracy_difference']:+.1f} points "
            f"(95% bootstrap CI {100 * low:+.1f} to {100 * high:+.1f}; "
            f"exact McNemar p = {result['mcnemar_exact_p']:.2g})"
        )
    audit = rows["qcv"]["separation_audit"]
    if audit is not None:
        lines.append(
            f"Eq. (8) audit: kappa < 1 on {100 * audit['sufficient_condition_rate']:.1f}% of "
            f"{audit['examples']} covered examples; wrong answers among them: "
            f"{audit['sufficient_condition_violations']} (the condition guarantees 0)"
        )
    lines.append(f"files written to {root}")
    report = "\n".join(lines)
    (root / "report.txt").write_text(report + "\n", encoding="utf-8")
    return report
