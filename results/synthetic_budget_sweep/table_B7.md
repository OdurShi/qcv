Budget B = 7; metrics after the per-input accuracies are macro-averages.

| Method | synthetic | Acc_all | Cov. | Acc_cov | Corr. | Corrupt. | rho_B | B_eff | S_B | Err. AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| target-only | 89.8 | 89.8 | 100.0 | 89.8 | 0.0 | 0.0 | - | - | - | - |
| repeated-target | 87.8 | 87.8 | 100.0 | 87.8 | 3.4 | 2.6 | 0.40 | 2.05 | 44.4 | - |
| fixed-probes | 97.0 | 97.0 | 100.0 | 97.0 | 82.9 | 1.3 | 0.08 | 4.82 | 69.8 | 92.9 |
| random-qcv | 96.1 | 96.1 | 100.0 | 96.1 | 79.0 | 1.9 | 0.05 | 5.31 | 74.3 | 89.8 |
| reliability-qcv | 97.2 | 97.2 | 100.0 | 97.2 | 83.4 | 1.3 | 0.01 | 6.63 | 86.9 | 91.7 |
| diagonal-qcv | 96.1 | 96.1 | 100.0 | 96.1 | 62.9 | 0.1 | 0.07 | 4.83 | 69.7 | 93.4 |
| qcv | 97.2 | 97.2 | 100.0 | 97.2 | 82.9 | 1.2 | 0.03 | 5.91 | 78.4 | 92.3 |
