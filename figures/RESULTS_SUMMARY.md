# Results summary (auto-generated)

## Research-style resume bullet
Built a validated, explainable heart-disease detection pipeline on 918 patients from four hospitals; nested CV found no statistically significant difference between logistic regression (0.88), Random Forest (0.88) and Gradient Boosting (0.88) AUC; leave-one-hospital-out AUC ranged 0.71-0.85; SHAP importances agreed across models (Spearman 0.76-0.92).

## Industry-style resume bullet
Built an interactive Streamlit risk-scoring demo for heart-disease detection with risk tiers, per-patient SHAP explanations and a cost-based decision threshold (assumed 5:1 miss-to-false-alarm cost; at an assumed 10% prevalence it flags 157 per 1,000 and misses 30.4 of 100 cases); audited subgroup and cross-hospital performance, documented calibration drift and a local recalibration fix (Brier -8% on average); added unit tests, CI and a model card.

Check every claim against the tables in MODEL_CARD.md before using these.
