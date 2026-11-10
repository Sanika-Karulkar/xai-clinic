"""Model running: the entry point.

    uv run xai-clinic --quick          # 2-3 min smoke test (rough numbers)
    uv run xai-clinic                  # full run, ~15-30 min
    uv run xai-clinic --cost-ratio 10 --prevalence 0.05 --model "Logistic Regression"

Writes every figure, table, the model card and the app's model file into ./figures
"""
import argparse
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import data_processing as dp
from . import decision as dec
from . import explain as xai
from . import modeling as mdl
from . import reporting as rep
from .config import FIG_DIR, MODEL_NAMES, SITES, TIER_CUTS, Settings
from .data_loading import load_data


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="xai-clinic", description="Run the full XAI Clinic analysis.")
    p.add_argument("--quick", action="store_true", help="fast smoke test (rough numbers)")
    p.add_argument("--model", choices=MODEL_NAMES, default=None,
                   help="force the shipped model (default: best nested-CV mean)")
    p.add_argument("--cost-ratio", type=float, default=5.0,
                   help="cost of a missed patient in false-alarm units (default 5)")
    p.add_argument("--prevalence", type=float, default=0.10,
                   help="disease rate of the population you would screen (default 0.10)")
    p.add_argument("--primary-flags", action="store_true",
                   help="make the missing-flag model the primary one (default: hospital-agnostic)")
    return p.parse_args(argv)


def run(s: Settings):
    warnings.filterwarnings("ignore")
    R = {"settings": s}

    # ------------------------------------------------ 1. data + missingness
    df = load_data()
    df, n_dup = dp.deduplicate(df)
    print(f"Exact duplicate rows removed: {n_dup}")
    print("\nPatients per hospital:\n", df["site"].value_counts().to_string())
    prev_by_site = df.groupby("site")["target"].mean().round(2)
    print("\nDisease rate per hospital:\n", prev_by_site.to_string())
    prev_total = df["target"].mean()
    print(f"Total patients: {len(df)}  (overall disease rate {prev_total:.0%})")

    overall, by_site = dp.missingness(df)
    print("\nMissing values (%):\n", overall.round(1).to_string())
    print("\n% missing by hospital:\n", by_site.T.to_string())
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    by_site.T.to_csv(f"{FIG_DIR}/missing_by_site.csv")
    rep.plot_missing(overall, by_site, s.missing_threshold)
    site_table = pd.DataFrame({"patients": df.groupby("site").size(), "disease rate": prev_by_site})

    df, dropped = dp.drop_sparse_features(df, s.missing_threshold)
    print("Dropped (too much missing):", dropped)
    ds = dp.build_dataset(df)
    print(f"Sparse records (>=3 core values missing): {int(ds.sparse.sum())} "
          f"-> by hospital: {ds.sparse.groupby(ds.site).sum().to_dict()}")
    X_train, X_test, y_train, y_test = dp.split_dataset(ds, s.seed)
    print(f"\nTrain n={len(X_train)}, test n={len(X_test)}")
    R.update(n_total=len(df), n_dup=n_dup, prev_total=prev_total, site_table=site_table,
             dropped=dropped, final_cols=ds.final_cols, n_test=len(y_test))

    # ------------------------------------------------------ 2. ablation
    print("\n=== ABLATION (pooled repeated 5-fold CV, default settings, ROC-AUC) ===")
    abl = mdl.run_ablation(X_train, y_train, ds.raw_cols, s)
    print(abl.to_string(index=False))
    abl.to_csv(f"{FIG_DIR}/ablation.csv", index=False)
    print("NOTE: a gain from missing-flags in POOLED CV can be a hospital shortcut; "
          "the leave-one-hospital-out comparison further down decides.")

    # ------------------------------------------------- 3. nested CV
    print("\n=== NESTED CV (tuning inside every outer fold) ===")
    nested = mdl.nested_cv(X_train, y_train, ds.final_cols, s)
    pair_df = mdl.paired_tests(nested)
    print("\nPaired comparisons (corrected t-test):\n", pair_df.to_string(index=False))
    pair_df.to_csv(f"{FIG_DIR}/paired_tests.csv", index=False)
    nested_df = pd.DataFrame(nested)
    nested_df.to_csv(f"{FIG_DIR}/nested_cv_scores.csv", index=False)
    rep.plot_nested(nested_df)
    best_name, all_tied = mdl.choose_best(nested, pair_df, s)
    print(f"\nSelected model: {best_name}. "
          + ("No pair differs significantly -> models are statistically TIED." if all_tied
             else "At least one pair differs significantly."))
    R.update(nested=nested, all_tied=all_tied, best_name=best_name)

    # --------------------------------------- 4. tuning + held-out test
    tuned = mdl.tune_all(X_train, y_train, ds.final_cols, s)
    best = tuned[best_name]
    print(f"\n=== HELD-OUT TEST SET (n={len(y_test)}), threshold 0.5, 95% bootstrap CI ===")
    test_tbl, test_probs = mdl.test_table(tuned, X_test, y_test, s)
    print(test_tbl.to_string())
    print(f"(Brier of a no-skill model that always predicts the prevalence: {prev_total * (1 - prev_total):.3f})")
    test_tbl.to_csv(f"{FIG_DIR}/test_results_with_CI.csv")
    p_best = test_probs[best_name]
    rep.plot_performance(tuned, best, best_name, X_test, y_test, p_best)
    R["test_tbl"] = test_tbl

    # ---------------------------------------------- 5. decision analysis
    print(f"\n=== DECISION ANALYSIS (a missed patient costs {s.cost_fn_to_fp:g}x a false alarm; "
          f"deployment prevalence {s.deploy_prevalence:.0%}) ===")
    cv5 = StratifiedKFold(5, shuffle=True, random_state=s.seed)
    oof = cross_val_predict(best, X_train, y_train, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = dec.choose_thresholds(oof, y_train, s)
    print("Thresholds chosen on TRAIN out-of-fold predictions only:",
          {k: round(v, 2) for k, v in thresholds.items()})
    op_df = dec.operating_points(y_test, p_best, thresholds, s)
    print(f"\nImpact per 1,000 screened patients at {s.deploy_prevalence:.0%} prevalence:")
    print(op_df.to_string(index=False))
    print("(PPV falls sharply at low prevalence - the same model is far less 'precise' in a screening population.)")
    op_df.to_csv(f"{FIG_DIR}/operating_points.csv", index=False)
    rep.plot_decision_curve(y_test, p_best, best_name)
    R["op_df"] = op_df

    # ------------------------------------------- 6. tiers + subgroups
    oof_all = cross_val_predict(best, ds.X, ds.y, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]
    tier_df = dec.tier_table(oof_all, ds.y)
    print(f"\nRisk tiers (cut-offs {TIER_CUTS}, out-of-fold predictions on all patients):\n", tier_df.to_string())
    tier_df.to_csv(f"{FIG_DIR}/risk_tiers.csv")
    sub_df = mdl.subgroup_table(ds, oof_all)
    print("\nSubgroup performance (5-fold out-of-fold predictions):\n", sub_df.to_string(index=False))
    print("(hyper-parameters were tuned on the training split, so these are slightly optimistic)")
    sub_df.to_csv(f"{FIG_DIR}/subgroup_performance.csv", index=False)
    R.update(tier_df=tier_df, sub_df=sub_df)

    # ------------------------- 7. leave-one-hospital-out + recalibration
    print("\n=== LEAVE-ONE-HOSPITAL-OUT (re-tuned without the held-out hospital) ===")
    loso, loso_preds = mdl.leave_one_hospital_out(ds, ds.final_cols, s)
    loso.to_csv(f"{FIG_DIR}/leave_one_hospital_out.csv", index=False)
    piv = loso.pivot_table(index="hospital", columns=["model", "missing-flags"], values="AUC")
    print(piv.round(3).to_string())
    mean_flag = loso.groupby("missing-flags")["AUC"].mean().round(3)
    print(f"\nMean leave-one-hospital-out AUC  without flags: {mean_flag.get('no', np.nan)}  |  "
          f"with flags: {mean_flag.get('yes', np.nan)}")
    cells = loso.pivot_table(index=["hospital", "model"], columns="missing-flags", values="AUC")
    flag_gain = float(mean_flag.get("yes", np.nan) - mean_flag.get("no", np.nan))
    n_better, n_cells = int((cells["yes"] > cells["no"]).sum()), len(cells)
    if flag_gain > 0.01:
        print(f"-> Missing-flags raise AUC at UNSEEN hospitals by {flag_gain:+.3f} on average "
              f"(better in {n_better}/{n_cells} model-hospital cells). So missingness carries transferable signal "
              "(clinical workflow, and hospital-level baseline differences), not only a shortcut.\n"
              "   Report both versions; the hospital-agnostic PRIMARY model stays the default for deployability "
              "unless you can justify depending on which tests were recorded (use --primary-flags).")
    else:
        print(f"-> Flags change AUC at unseen hospitals by only {flag_gain:+.3f} (better in {n_better}/{n_cells} cells): "
              "the pooled-CV gain was mostly a hospital shortcut, so the hospital-agnostic PRIMARY model is the right one to ship.")
    cal_rows = loso[(loso.model == best_name) & (loso["missing-flags"] == "no")]
    print(f"\nCalibration-in-the-large, {best_name}:\n",
          cal_rows[["hospital", "actual_prev", "mean_pred", "Brier"]].to_string(index=False))

    print("\nAUC on dense vs sparse records within each held-out hospital:")
    for (name, h), (yt, p, idx) in loso_preds.items():
        if name != best_name:
            continue
        sp = ds.sparse.values[idx]
        out = []
        for lab, m in [("dense", ~sp), ("sparse", sp)]:
            pos = int(yt[m].sum()); neg = int(m.sum() - pos)
            out.append(f"{lab}: n={int(m.sum())} AUC=" +
                       (f"{roc_auc_score(yt[m], p[m]):.3f}" if min(pos, neg) >= 5 else "n/a"))
        print(f"  {h:10s} " + " | ".join(out))
    rep.plot_calibration_by_hospital(loso_preds, best_name)

    print("\n=== LOCAL RECALIBRATION: use 30% of the new hospital's patients to re-fit intercept+slope ===")
    recal_df, recal_gain = mdl.recalibration_demo(loso_preds, best_name, s)
    print(recal_df.to_string(index=False))
    print("(AUC is unchanged by recalibration; only the absolute risk estimates improve.)")
    recal_df.to_csv(f"{FIG_DIR}/local_recalibration.csv", index=False)
    R.update(piv=piv, mean_flag=mean_flag, n_better=n_better, n_cells=n_cells, recal_df=recal_df,
             recal_gain=recal_gain, loso_best=cal_rows.set_index("hospital")["AUC"])

    # ------------------------------------------------------ 8. SHAP
    sv_all, imp_df = xai.importance_by_model(tuned, X_train, X_test)
    print("\n=== DO THE THREE MODELS TELL THE SAME STORY? (grouped by clinical variable) ===")
    rho, overlaps = xai.agreement(imp_df)
    print(rho.to_string())
    for k, v in overlaps.items():
        print(f"  top-5 overlap {k}: {v}/5")
    rho.to_csv(f"{FIG_DIR}/shap_rank_agreement.csv")
    imp_df.sort_values(best_name, ascending=False).round(4).to_csv(f"{FIG_DIR}/shap_importance_grouped.csv")
    rep.plot_shap_agreement(imp_df)
    rep.plot_shap_details(best_name, sv_all[best_name], best[:-1].transform(X_test), p_best)
    R["rho"] = rho

    # ------------------------------------------------- 9. deliverables
    joblib.dump({
        "pipeline": best, "model_name": best_name,
        "input_columns": list(ds.X.columns), "feature_cols": ds.final_cols,
        "background": X_train.sample(min(100, len(X_train)), random_state=s.seed),
        "thresholds": thresholds, "tier_cuts": TIER_CUTS,
        "metrics": {"test": test_tbl.loc[best_name].to_dict(), "nested_cv_auc": float(nested[best_name].mean()),
                    "n_patients": int(len(df)), "built": str(date.today())},
    }, f"{FIG_DIR}/model_artifact.joblib")
    rep.write_environment(s)
    rep.write_model_card(R)
    research, industry = rep.write_summary(R)
    print("\n" + "=" * 70 + "\nDRAFT RESUME BULLETS (also saved to figures/RESULTS_SUMMARY.md)\n" + "=" * 70)
    print("RESEARCH:", research, "\n\nINDUSTRY:", industry)
    print(f"\nDone. Everything is in ./{FIG_DIR}  (open MODEL_CARD.md first)")
    return R


def main(argv=None):
    a = parse_args(argv)
    run(Settings(quick=a.quick, force_model=a.model, cost_fn_to_fp=a.cost_ratio,
                 deploy_prevalence=a.prevalence, primary_flags=a.primary_flags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
