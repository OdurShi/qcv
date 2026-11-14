Budget B = 5; metrics after the per-input accuracies are macro-averages.

| Method | synthetic | Acc_all | Cov. | Acc_cov | Corr. | Corrupt. | rho_B | B_eff | S_B | Err. AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| target-only | 89.8 | 89.8 | 100.0 | 89.8 | 0.0 | 0.0 | - | - | - | - |
| repeated-target | 87.4 | 87.4 | 100.0 | 87.4 | 3.4 | 3.0 | 0.39 | 1.94 | 45.0 | - |
| fixed-probes | 96.8 | 96.8 | 100.0 | 96.8 | 77.1 | 0.9 | 0.06 | 4.03 | 73.2 | 93.3 |
| random-qcv | 94.1 | 94.1 | 100.0 | 94.1 | 63.4 | 2.4 | 0.07 | 3.88 | 74.5 | 86.4 |
| reliability-qcv | 91.5 | 91.5 | 100.0 | 91.5 | 56.1 | 4.5 | 0.01 | 4.83 | 89.1 | 86.7 |
| diagonal-qcv | 93.3 | 93.3 | 100.0 | 93.3 | 35.1 | 0.1 | 0.16 | 3.03 | 59.8 | 89.4 |
| qcv | 96.9 | 96.9 | 100.0 | 96.9 | 83.9 | 1.6 | 0.01 | 4.83 | 85.6 | 93.2 |
