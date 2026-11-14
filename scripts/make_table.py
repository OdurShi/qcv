"""Aggregate sweep outputs into the layout of the paper's result tables.

Each input is a ``sweep.csv`` written by ``run_ablation_sweep.py`` for one
backbone x benchmark configuration. For a given budget the script prints one
Markdown row per method with every metric macro-averaged over the inputs
(the paper macro-averages its four benchmarks), plus the per-input accuracies.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

COLUMNS = (
    ("overall_accuracy", "Acc_all", 100.0),
    ("coverage", "Cov.", 100.0),
    ("covered_accuracy", "Acc_cov", 100.0),
    ("correction_rate", "Corr.", 100.0),
    ("corruption_rate", "Corrupt.", 100.0),
    ("mean_error_correlation", "rho_B", 1.0),
    ("effective_measurements", "B_eff", 1.0),
    ("correct_measurement_survival", "S_B", 100.0),
    ("margin_error_auroc", "Err. AUROC", 100.0),
)


def read_rows(path: Path, budget: int) -> dict[str, dict[str, float | None]]:
    """Metrics of every method at ``budget``."""
    rows: dict[str, dict[str, float | None]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["budget"]) != budget:
                continue
            rows[row["method"]] = {
                key: (float(row[key]) if row.get(key) not in (None, "") else None)
                for key, _, _ in COLUMNS
            }
    return rows


def macro_average(tables: list[dict[str, dict[str, float | None]]]) -> dict[str, dict[str, float]]:
    """Mean of each metric over the inputs that report it, per method."""
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for table in tables:
        for method, metrics in table.items():
            for key, value in metrics.items():
                if value is not None:
                    collected[method][key].append(value)
    return {
        method: {key: sum(values) / len(values) for key, values in metrics.items()}
        for method, metrics in collected.items()
    }


def format_table(
    names: list[str], tables: list[dict[str, dict[str, float | None]]], budget: int
) -> str:
    averages = macro_average(tables)
    header = ["Method", *names, *(title for _, title, _ in COLUMNS)]
    lines = [
        f"Budget B = {budget}; metrics after the per-input accuracies are macro-averages.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for method, metrics in averages.items():
        cells = [method]
        for table in tables:
            value = table.get(method, {}).get("overall_accuracy")
            cells.append("-" if value is None else f"{100 * value:.1f}")
        for key, _, scale in COLUMNS:
            value = metrics.get(key)
            if value is None:
                cells.append("-")
            else:
                cells.append(f"{value:.2f}" if scale == 1.0 else f"{value * scale:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweeps", nargs="+", help="sweep.csv files, one per configuration")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument(
        "--names", nargs="+", help="column label of each input (default: parent directory names)"
    )
    args = parser.parse_args()
    paths = [Path(item) for item in args.sweeps]
    names = args.names or [path.parent.name for path in paths]
    if len(names) != len(paths):
        parser.error("--names must have one label per sweep file")
    print(format_table(names, [read_rows(path, args.budget) for path in paths], args.budget))


if __name__ == "__main__":
    main()
