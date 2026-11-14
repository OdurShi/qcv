Budget B = 3; metrics after the per-input accuracies are macro-averages.

| Method | synthetic | Acc_all | Cov. | Acc_cov | Corr. | Corrupt. | rho_B | B_eff | S_B | Err. AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| target-only | 89.8 | 89.8 | 100.0 | 89.8 | 0.0 | 0.0 | - | - | - | - |
| repeated-target | 86.0 | 86.0 | 100.0 | 86.0 | 2.9 | 4.5 | 0.38 | 1.71 | 46.2 | - |
| fixed-probes | 90.0 | 90.0 | 100.0 | 90.0 | 18.0 | 1.7 | -0.02 | 3.00 | 91.0 | 86.7 |
| random-qcv | 90.6 | 90.6 | 100.0 | 90.6 | 29.8 | 2.4 | 0.13 | 2.40 | 73.7 | 79.8 |
| reliability-qcv | 88.3 | 88.3 | 100.0 | 88.3 | 29.3 | 4.9 | 0.00 | 2.98 | 87.8 | 78.9 |
| diagonal-qcv | 92.3 | 92.3 | 100.0 | 92.3 | 27.8 | 0.3 | 0.23 | 2.05 | 55.4 | 92.6 |
| qcv | 93.2 | 93.2 | 100.0 | 93.2 | 71.2 | 4.3 | 0.07 | 2.64 | 80.0 | 89.0 |
