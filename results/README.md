# Results of the synthetic configurations

Raw outputs of runs made with this repository on 2026-11-13 (Python 3.11.18, package versions as in `requirements-dev.lock`). All runs use the synthetic scenes in `data/synthetic/` and the simulated VLM of `qcv/synthetic.py`; they are not runs of the paper's benchmarks.

| Folder | Command | Files |
|---|---|---|
| `synthetic_demo/default/` | `qcv demo --output-dir runs/demo` | `report.txt`, `<method>.metrics.json`, `qcv_vs_<method>.json` |
| `synthetic_demo/independent/` | `qcv demo --output-dir runs/demo_independent --correlation 0.0` | as above |
| `synthetic_demo/hard/` | `qcv demo --output-dir runs/demo_hard --error-rate 0.3` | as above |
| `synthetic_demo/hard_independent/` | `qcv demo --output-dir runs/demo_hard_independent --error-rate 0.3 --correlation 0.0` | as above |
| `synthetic_budget_sweep/` | `qcv calibrate --input runs/demo/calibration_scores.jsonl --output-dir runs/demo/artifacts_b9 --max-budget 9`, then `python scripts/run_ablation_sweep.py --config configs/budget_sweep.yaml --scores runs/demo/test_scores.jsonl --artifacts runs/demo/artifacts_b9 --output-dir runs/demo/sweep`, then `python scripts/make_table.py runs/demo/sweep/sweep.csv --budget <B> --names synthetic` | `sweep.csv`, `comparisons.json`, `correlations.json`, `table_B<B>.md`, `<method>.B<B>.metrics.json`, `sweep_log.txt` |

`report.txt` is the text printed by `qcv demo`. Metrics files are the output of `qcv evaluate`; `qcv_vs_<method>.json` and `comparisons.json` hold the paired bootstrap interval and the exact McNemar test of QCV against the other method. Prediction, score and calibration files are not stored; the commands regenerate them.
