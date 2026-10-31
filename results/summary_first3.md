Run: first3 | patients: 3 | trials: 120 | leave-one-patient-out

| Model | Mean balanced acc. | SD across patients | Patients above chance |
|---|---|---|---|
| CSP + LDA | 0.517 | 0.052 | 0/3 |
| CSP + LogReg | 0.525 | 0.043 | 0/3 |
| CSP + Linear SVM | 0.500 | 0.025 | 0/3 |
| Covariances + Riemannian + LogReg | 0.492 | 0.014 | 0/3 |

Frequency bands (CSP + LDA):
- mu 8–13: 0.533
- beta 13–30: 0.508
- combined 8–30: 0.517