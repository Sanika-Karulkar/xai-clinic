"""Reporting: figures, the auto-generated model card, resume-bullet drafts, environment log."""
import os
import platform
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import FIG_DIR, MODEL_NAMES, SITES, TIER_CUTS
from .data_processing import group_name, pretty
from .decision import net_benefit


def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(f"{FIG_DIR}/{name}", dpi=200, bbox_inches="tight")
    plt.close("all")


def md_table(d, index=True):
    d = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [" / ".join(map(str, c)) for c in d.columns]
    d = d.rename_axis(d.index.name or "model").reset_index() if index else d
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ figures
def plot_missing(overall, by_site, threshold):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=overall.values, y=overall.index, color="steelblue")
    plt.axvline(threshold * 100, color="red", ls="--", label="drop threshold")
    plt.xlabel("% missing"); plt.title("Missing data by feature"); plt.legend()
    savefig("01_missing_data.png")
    plt.figure(figsize=(7, 5))
    sns.heatmap(by_site.T, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label": "% missing"})
    plt.title("Missingness is a hospital fingerprint")
    savefig("01b_missing_by_hospital.png")


def plot_nested(nested_df):
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=nested_df, color="lightsteelblue")
    sns.stripplot(data=nested_df, color="black", size=4)
    plt.ylabel("ROC-AUC (outer fold)"); plt.title("Nested CV: per-fold AUC")
    savefig("03_nested_cv.png")


def plot_performance(tuned, best, best_name, X_test, y_test, p_best):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for name, m in tuned.items():
        RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=ax[0])
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4); ax[0].set_title("ROC curves (test set)")
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax[1], cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax[1].set_title(f"Confusion matrix - {best_name}")
    f, m = calibration_curve(y_test, p_best, n_bins=6, strategy="quantile")
    ax[2].plot(m, f, "o-", label=best_name); ax[2].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[2].set_xlabel("Predicted probability"); ax[2].set_ylabel("Observed fraction with disease")
    ax[2].set_title("Calibration (test set)"); ax[2].legend()
    plt.tight_layout(); savefig("04_model_performance.png")


def plot_decision_curve(y_test, p_best, best_name):
    yt = np.asarray(y_test)
    pts = np.linspace(0.05, 0.90, 60)
    prev = yt.mean()
    plt.figure(figsize=(7, 5))
    plt.plot(pts, [net_benefit(yt, p_best, t) for t in pts], label=best_name, lw=2)
    plt.plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], "--", label="flag everyone")
    plt.axhline(0, color="k", lw=1, label="flag no one")
    plt.ylim(-0.1, max(0.5, prev + 0.05))
    plt.xlabel("Threshold probability (risk at which you would act)")
    plt.ylabel("Net benefit"); plt.title("Decision curve (test set)"); plt.legend()
    savefig("05_decision_curve.png")


def plot_calibration_by_hospital(loso_preds, best_name):
    plt.figure(figsize=(6, 5.5))
    for s in SITES:
        if (best_name, s) in loso_preds:
            yt, p, _ = loso_preds[(best_name, s)]
            f, m = calibration_curve(yt, p, n_bins=5, strategy="quantile")
            plt.plot(m, f, "o-", label=s)
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed fraction with disease")
    plt.title(f"Calibration on an unseen hospital - {best_name}"); plt.legend()
    savefig("06_calibration_by_hospital.png")


def plot_shap_agreement(imp_df):
    order = imp_df.mean(axis=1).nlargest(10).index
    imp_df.loc[order[::-1]].plot.barh(figsize=(8, 6))
    plt.xlabel("Share of total mean |SHAP| (dummies summed per variable)")
    plt.title("What drives predictions - 3 models side by side")
    plt.tight_layout(); savefig("07_shap_model_agreement.png")


def plot_shap_details(best_name, sv, A, p_best):
    import shap
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title(f"SHAP summary ({best_name}): how each feature pushes risk up or down")
    savefig("08_shap_beeswarm.png")
    numeric = [c for c in A.columns if group_name(c) == c]
    top2 = pd.Series(np.abs(sv.values).mean(0), index=A.columns)[numeric].nlargest(2).index
    for i, col in enumerate(top2):
        shap.plots.scatter(sv[:, list(A.columns).index(col)], color=sv, show=False)
        savefig(f"09_shap_dependence_{i + 1}.png")
    for tag, j in [("highest_risk", p_best.argmax()), ("lowest_risk", p_best.argmin())]:
        shap.plots.waterfall(sv[j], max_display=10, show=False)
        plt.title(f"{tag.replace('_', ' ').title()} patient - predicted risk {p_best[j]:.0%}")
        savefig(f"10_waterfall_{tag}.png")


# --------------------------------------------------------------- documents
def write_environment(settings):
    import shap, sklearn
    with open(f"{FIG_DIR}/environment.txt", "w") as f:
        f.write(f"python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\n"
                f"scikit-learn {sklearn.__version__}\nshap {shap.__version__}\n"
                f"seed {settings.seed}\nquick_mode {settings.quick}\n")


def write_model_card(R):
    s, best_name = R["settings"], R["best_name"]
    nm = {n: R["nested"][n].mean() for n in MODEL_NAMES}
    loso_txt = ", ".join(f"{h} {v:.2f}" for h, v in R["loso_best"].items())
    tied = ("no significant difference between the models (all corrected p > 0.05)" if R["all_tied"]
            else "at least one significant difference between models (see paired_tests.csv)")
    mf = R["mean_flag"]
    card = f"""# Model card - XAI Clinic heart-disease detection ({best_name})

**Built:** {date.today()}  |  **Data:** UCI Heart Disease, {R['n_total']} patients, 4 hospitals  |  **Seed:** {s.seed}

## Intended use
Educational / research demonstration of an explainable risk-ranking model. **Not a medical device and
not validated for clinical use.** Outputs are risk scores for presence of coronary disease at testing,
not predictions of future cardiovascular events.

## Data
{md_table(R['site_table'])}

{R['n_dup']} exact duplicates removed. Dropped (>{s.missing_threshold:.0%} missing): {', '.join(R['dropped']) or 'none'}.
Remaining gaps imputed inside the cross-validation pipeline. The PRIMARY model sees no missing-flags (hospital-agnostic).

## Model
{best_name}, features: {', '.join(pretty(c) for c in R['final_cols'])}. Selected by nested-CV mean AUC; result: {tied}.

## Performance
Nested-CV AUC (train split): {', '.join(f'{n} {v:.3f}' for n, v in nm.items())}.

Held-out test set (n={R['n_test']}), threshold 0.5, 95% bootstrap CI:
{md_table(R['test_tbl'])}

Leave-one-hospital-out AUC ({best_name}, re-tuned without the hospital): {loso_txt}.
Missing-flags sensitivity run: mean AUC at unseen hospitals {mf.get('no', float('nan')):.3f} without flags vs {mf.get('yes', float('nan')):.3f} with flags ({R['n_better']}/{R['n_cells']} model-hospital cells improved).

{md_table(R['piv'].round(3))}

## Operating points (per 1,000 patients at {s.deploy_prevalence:.0%} prevalence; missed case = {s.cost_fn_to_fp:g}x false alarm)
{md_table(R['op_df'], index=False)}

## Risk tiers (out-of-fold, cut-offs {TIER_CUTS})
{md_table(R['tier_df'])}

## Subgroups
{md_table(R['sub_df'], index=False)}

## Calibration and transfer
Predicted risk drifts toward the pooled prevalence at a new hospital. Local recalibration (30% of the
new hospital's data) changes Brier as follows:
{md_table(R['recal_df'], index=False)}

## Explainability
SHAP importance grouped by clinical variable; Spearman rank agreement across the three models:
{md_table(R['rho'])}

## Limitations
- 1980s data, 4 hospitals, mostly male; women nearly absent from two hospitals.
- Labels = angiographic disease at testing; patients were referred for testing (referral bias, e.g. asymptomatic chest pain ranks high).
- SHAP explains the model, not causes of disease.
- Pooled prevalence is ~{R['prev_total']:.0%}; real screening populations are far lower, so PPV will be much lower than in this data.
- Cost ratio and deployment prevalence in the decision analysis are assumptions, not measurements.
"""
    open(f"{FIG_DIR}/MODEL_CARD.md", "w", encoding="utf-8").write(card)


def write_summary(R):
    s = R["settings"]
    nm = {n: R["nested"][n].mean() for n in MODEL_NAMES}
    rho, loso_best = R["rho"], R["loso_best"]
    cmp_txt = (f"nested CV found no statistically significant difference between logistic regression "
               f"({nm['Logistic Regression']:.2f}), Random Forest ({nm['Random Forest']:.2f}) and Gradient Boosting "
               f"({nm['Gradient Boosting']:.2f}) AUC" if R["all_tied"] else
               f"nested CV AUC was {nm['Logistic Regression']:.2f} (logistic regression), {nm['Random Forest']:.2f} "
               f"(Random Forest) and {nm['Gradient Boosting']:.2f} (Gradient Boosting), with at least one significant difference")
    tri = rho.values[np.triu_indices(3, 1)]
    research = (f"Built a validated, explainable heart-disease detection pipeline on {R['n_total']} patients from four hospitals; "
                f"{cmp_txt}; leave-one-hospital-out AUC ranged {loso_best.min():.2f}-{loso_best.max():.2f}; "
                f"SHAP importances agreed across models (Spearman {tri.min():.2f}-{tri.max():.2f}).")
    row = R["op_df"][R["op_df"]["operating point"] == "cost-optimal"].iloc[0]
    industry = (f"Built an interactive Streamlit risk-scoring demo for heart-disease detection with risk tiers, per-patient SHAP "
                f"explanations and a cost-based decision threshold (assumed {s.cost_fn_to_fp:g}:1 miss-to-false-alarm cost; at an assumed "
                f"{s.deploy_prevalence:.0%} prevalence it flags {row['flagged /1000']:.0f} per 1,000 and misses "
                f"{row['cases missed']:.1f} of {1000 * s.deploy_prevalence:.0f} cases); audited subgroup and cross-hospital performance, "
                f"documented calibration drift and a local recalibration fix (Brier -{np.mean(R['recal_gain']):.0%} on average); "
                f"added unit tests, CI and a model card.")
    open(f"{FIG_DIR}/RESULTS_SUMMARY.md", "w", encoding="utf-8").write(
        f"# Results summary (auto-generated)\n\n## Research-style resume bullet\n{research}\n\n"
        f"## Industry-style resume bullet\n{industry}\n\n"
        "Check every claim against the tables in MODEL_CARD.md before using these.\n")
    return research, industry
