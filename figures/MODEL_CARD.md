# Model card - XAI Clinic heart-disease detection (Gradient Boosting)

**Built:** 2026-11-10  |  **Data:** UCI Heart Disease, 918 patients, 4 hospitals  |  **Seed:** 42

## Intended use
Educational / research demonstration of an explainable risk-ranking model. **Not a medical device and
not validated for clinical use.** Outputs are risk scores for presence of coronary disease at testing,
not predictions of future cardiovascular events.

## Data
| site | patients | disease rate |
|---|---|---|
| cleveland | 303 | 0.46 |
| hungarian | 293 | 0.36 |
| switzerland | 123 | 0.93 |
| va | 199 | 0.74 |

2 exact duplicates removed. Dropped (>60% missing): ca.
Remaining gaps imputed inside the cross-validation pipeline. The PRIMARY model sees no missing-flags (hospital-agnostic).

## Model
Gradient Boosting, features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar > 120, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, thalassemia test, heart-rate reserve, cholesterol / age. Selected by nested-CV mean AUC; result: no significant difference between the models (all corrected p > 0.05).

## Performance
Nested-CV AUC (train split): Logistic Regression 0.882, Random Forest 0.880, Gradient Boosting 0.885.

Held-out test set (n=184), threshold 0.5, 95% bootstrap CI:
| model | AUC | Accuracy | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 [0.826, 0.928] | 0.793 [0.734, 0.848] | 0.824 [0.748, 0.896] | 0.756 [0.659, 0.846] | 0.143 |
| Random Forest | 0.876 [0.822, 0.923] | 0.788 [0.728, 0.848] | 0.853 [0.784, 0.923] | 0.707 [0.608, 0.800] | 0.142 |
| Gradient Boosting | 0.880 [0.829, 0.926] | 0.788 [0.728, 0.843] | 0.833 [0.764, 0.906] | 0.732 [0.635, 0.824] | 0.14 |

Leave-one-hospital-out AUC (Gradient Boosting, re-tuned without the hospital): cleveland 0.84, hungarian 0.85, va 0.71.
Missing-flags sensitivity run: mean AUC at unseen hospitals 0.802 without flags vs 0.826 with flags (9/9 model-hospital cells improved).

| hospital | Gradient Boosting / no | Gradient Boosting / yes | Logistic Regression / no | Logistic Regression / yes | Random Forest / no | Random Forest / yes |
|---|---|---|---|---|---|---|
| cleveland | 0.841 | 0.85 | 0.826 | 0.886 | 0.84 | 0.869 |
| hungarian | 0.852 | 0.882 | 0.876 | 0.896 | 0.851 | 0.882 |
| va | 0.706 | 0.711 | 0.729 | 0.737 | 0.698 | 0.719 |

## Operating points (per 1,000 patients at 10% prevalence; missed case = 5x false alarm)
| operating point | threshold | sensitivity | specificity | flagged /1000 | cases caught | cases missed | false alarms | PPV | NPV | cost /1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| default (0.50) | 0.5 | 0.833 | 0.732 | 325 | 83.3 | 16.7 | 241 | 0.257 | 0.9753 | 325 |
| 90%-sensitivity | 0.4 | 0.902 | 0.683 | 376 | 90.2 | 9.8 | 285 | 0.24 | 0.9843 | 334 |
| cost-optimal | 0.7 | 0.696 | 0.902 | 157 | 69.6 | 30.4 | 88 | 0.442 | 0.9639 | 240 |

## Risk tiers (out-of-fold, cut-offs (0.3, 0.7))
| tier | patients | observed_rate | mean_predicted |
|---|---|---|---|
| Low | 268 | 0.108 | 0.147 |
| Moderate | 257 | 0.525 | 0.514 |
| High | 393 | 0.875 | 0.854 |

## Subgroups
| subgroup | n | disease rate | AUC |
|---|---|---|---|
| all patients | 918 | 0.55 | 0.879 |
| female | 193 | 0.26 | 0.856 |
| male | 725 | 0.63 | 0.859 |
| age < 50 | 291 | 0.38 | 0.898 |
| age 50-59 | 374 | 0.57 | 0.866 |
| age 60+ | 253 | 0.73 | 0.807 |
| hospital: cleveland | 303 | 0.46 | 0.886 |
| hospital: hungarian | 293 | 0.36 | 0.877 |
| hospital: switzerland | 123 | 0.93 | n/a (too few) |
| hospital: va | 199 | 0.74 | 0.72 |
| sparse records (3+ core values missing) | 55 | 0.64 | 0.726 |
| dense records | 863 | 0.55 | 0.885 |

## Calibration and transfer
Predicted risk drifts toward the pooled prevalence at a new hospital. Local recalibration (30% of the
new hospital's data) changes Brier as follows:
| hospital | Brier before | Brier after | improvement |
|---|---|---|---|
| cleveland | 0.172 | 0.165 | 4% |
| hungarian | 0.165 | 0.151 | 8% |
| va | 0.2 | 0.177 | 12% |

## Explainability
SHAP importance grouped by clinical variable; Spearman rank agreement across the three models:
| model | Logistic Regression | Random Forest | Gradient Boosting |
|---|---|---|---|
| Logistic Regression | 1.0 | 0.81 | 0.76 |
| Random Forest | 0.81 | 1.0 | 0.92 |
| Gradient Boosting | 0.76 | 0.92 | 1.0 |

## Limitations
- 1980s data, 4 hospitals, mostly male; women nearly absent from two hospitals.
- Labels = angiographic disease at testing; patients were referred for testing (referral bias, e.g. asymptomatic chest pain ranks high).
- SHAP explains the model, not causes of disease.
- Pooled prevalence is ~55%; real screening populations are far lower, so PPV will be much lower than in this data.
- Cost ratio and deployment prevalence in the decision analysis are assumptions, not measurements.
