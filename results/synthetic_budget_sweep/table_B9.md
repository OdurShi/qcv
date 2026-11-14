Budget B = 9; metrics after the per-input accuracies are macro-averages.

| Method | synthetic | Acc_all | Cov. | Acc_cov | Corr. | Corrupt. | rho_B | B_eff | S_B | Err. AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| target-only | 89.8 | 89.8 | 100.0 | 89.8 | 0.0 | 0.0 | - | - | - | - |
| repeated-target | 88.3 | 88.3 | 100.0 | 88.3 | 1.0 | 1.7 | 0.40 | 2.14 | 44.9 | - |
| fixed-probes | 96.2 | 96.2 | 75.1 | 97.6 | 73.2 | 1.2 | 0.05 | 6.61 | 74.1 | 96.2 |
| random-qcv | 96.2 | 96.2 | 75.1 | 97.6 | 73.2 | 1.2 | 0.05 | 6.61 | 74.1 | 96.2 |
| reliability-qcv | 96.2 | 96.2 | 75.1 | 97.6 | 73.2 | 1.2 | 0.05 | 6.61 | 74.1 | 96.2 |
| diagonal-qcv | 96.2 | 96.2 | 75.1 | 97.7 | 63.9 | 0.1 | 0.05 | 6.61 | 74.1 | 95.9 |
| qcv | 96.2 | 96.2 | 75.1 | 97.6 | 73.2 | 1.2 | 0.05 | 6.61 | 74.1 | 96.2 |
