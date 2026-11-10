"""Model running: search spaces, ablation, nested CV, tuning, test-set metrics with
bootstrap CIs, subgroup audit, leave-one-hospital-out and local recalibration."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (ParameterGrid, RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, StratifiedShuffleSplit, cross_val_score,
                                     cross_validate)

from .config import ENGINEERED_ALL, ENGINEERED_FINAL, MODEL_NAMES, SITES
from .data_processing import make_pipeline

GRIDS = {
    "Logistic Regression": {"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]},
    "Random Forest": {"model__n_estimators": [100, 200, 300],
                      "model__max_depth": [2, 3, 5, 8, None],
                      "model__min_samples_leaf": [1, 2, 4, 8, 16],
                      "model__max_features": ["sqrt", "log2", 0.5]},
    "Gradient Boosting": {"model__n_estimators": [50, 100, 200],
                          "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                          "model__max_depth": [1, 2, 3],
                          "model__subsample": [0.5, 0.7, 0.85],
                          "model__min_samples_leaf": [3, 5, 10, 20]},
}


def make_estimator(name, final_cols, flags, seed):
    if name == "Logistic Regression":
        return make_pipeline(LogisticRegression(max_iter=5000, random_state=seed),
                             final_cols, scale=True, flags=flags)
    if name == "Random Forest":
        return make_pipeline(RandomForestClassifier(random_state=seed), final_cols, flags=flags)
    return make_pipeline(GradientBoostingClassifier(random_state=seed), final_cols, flags=flags)


def make_search(name, final_cols, settings, n_jobs=1, flags=None):
    """Random search (3-fold inner CV, ROC-AUC). flags=None -> use the PRIMARY setting."""
    flags = settings.primary_flags if flags is None else flags
    grid = GRIDS[name]
    if settings.quick and "model__n_estimators" in grid:
        grid = {**grid, "model__n_estimators": [30, 60]}
    n_iter = min(settings.n_iter, len(ParameterGrid(grid)))
    return RandomizedSearchCV(
        make_estimator(name, final_cols, flags, settings.seed), grid, n_iter=n_iter,
        cv=StratifiedKFold(3, shuffle=True, random_state=settings.seed),
        scoring="roc_auc", random_state=settings.seed, n_jobs=n_jobs)


# ---------------------------------------------------------------- ablation
def run_ablation(X_train, y_train, raw_cols, settings):
    """Do engineered features / missing-flags help? (pooled repeated 5-fold CV, default settings)"""
    variants = {
        "raw features only":              (raw_cols, False),
        "raw + 2 engineered (PRIMARY)":   (raw_cols + ENGINEERED_FINAL, False),
        "raw + all 4 engineered":         (raw_cols + ENGINEERED_ALL, False),
        "PRIMARY + missing-flags":        (raw_cols + ENGINEERED_FINAL, True),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=settings.ablation_repeats,
                                 random_state=settings.seed)
    rows = []
    for vname, (cols, flags) in variants.items():
        row = {"variant": vname}
        for mname, model, scale in [
                ("Logistic Regression", LogisticRegression(max_iter=5000), True),
                ("Random Forest", RandomForestClassifier(settings.rf_trees, random_state=settings.seed), False)]:
            s = cross_val_score(make_pipeline(model, cols, scale=scale, flags=flags), X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
            row[mname] = f"{s.mean():.3f} +/- {s.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- nested CV
def nested_cv(X_train, y_train, final_cols, settings):
    """Tuning happens INSIDE every outer fold, so scores are not optimistic."""
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=settings.nested_repeats,
                                    random_state=settings.seed)      # identical splits for every model
    scores = {}
    for name in MODEL_NAMES:
        scores[name] = cross_validate(make_search(name, final_cols, settings), X_train, y_train,
                                      cv=outer, scoring="roc_auc", n_jobs=-1)["test_score"]
        print(f"  {name:20s} AUC = {scores[name].mean():.3f} +/- {scores[name].std():.3f}")
    return scores


def paired_tests(nested):
    """Nadeau-Bengio corrected resampled t-test (CV folds overlap, so a plain t-test is too optimistic)."""
    rows = []
    for a, b in [("Random Forest", "Logistic Regression"), ("Gradient Boosting", "Logistic Regression"),
                 ("Random Forest", "Gradient Boosting")]:
        d = nested[a] - nested[b]
        k = len(d)
        se = np.sqrt((1 / k + 1 / 4) * d.var(ddof=1))
        tcrit = stats.t.ppf(0.975, k - 1)
        if se == 0:                                          # identical scores: no evidence of a difference
            p = 1.0 if d.mean() == 0 else 0.0
        else:
            p = 2 * stats.t.sf(abs(d.mean() / se), k - 1)
        rows.append({"comparison": f"{a} - {b}", "mean AUC diff": round(d.mean(), 4),
                     "95% CI": f"[{d.mean() - tcrit * se:.3f}, {d.mean() + tcrit * se:.3f}]",
                     "p-value": round(p, 3)})
    return pd.DataFrame(rows)


def choose_best(nested, pair_df, settings):
    best = max(MODEL_NAMES, key=lambda n: nested[n].mean())
    all_tied = bool((pair_df["p-value"] > 0.05).all())
    if settings.force_model in MODEL_NAMES:
        best = settings.force_model
        print(f"--model is set -> using {best}")
    return best, all_tied


# ------------------------------------------------------ tuning + test set
def tune_all(X_train, y_train, final_cols, settings):
    tuned = {}
    for name in MODEL_NAMES:
        rs = make_search(name, final_cols, settings, n_jobs=-1).fit(X_train, y_train)
        tuned[name] = rs.best_estimator_
        print(f"{name}: best params = {rs.best_params_}")
        grid = GRIDS[name]
        edge = [k.replace("model__", "") for k, v in rs.best_params_.items()
                if all(isinstance(g, (int, float)) for g in grid[k]) and len(grid[k]) > 2
                and v in (min(grid[k]), max(grid[k]))]
        if edge:
            print(f"   (at the edge of the search grid: {edge} - consider widening)")
    return tuned


def boot_metrics(yt, p, thr=0.5, n_boot=2000, seed=42):
    """Point estimate [95% bootstrap CI] for AUC, accuracy, sensitivity, specificity."""
    yt, p = np.asarray(yt), np.asarray(p)

    def m(a, b):
        pred = b >= thr
        tp, tn = ((pred == 1) & (a == 1)).sum(), ((pred == 0) & (a == 0)).sum()
        fp, fn = ((pred == 1) & (a == 0)).sum(), ((pred == 0) & (a == 1)).sum()
        return {"AUC": roc_auc_score(a, b), "Accuracy": (tp + tn) / len(a),
                "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp)}

    rng = np.random.default_rng(seed)
    point, boots = m(yt, p), []
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) == 2:
            boots.append(m(yt[i], p[i]))
    ci = pd.DataFrame(boots).quantile([0.025, 0.975])
    return {k: f"{point[k]:.3f} [{ci.loc[0.025, k]:.3f}, {ci.loc[0.975, k]:.3f}]" for k in point}


def test_table(tuned, X_test, y_test, settings):
    probs = {n: m.predict_proba(X_test)[:, 1] for n, m in tuned.items()}
    tbl = pd.DataFrame({n: boot_metrics(y_test, p, 0.5, settings.n_boot, settings.seed)
                        for n, p in probs.items()}).T
    tbl["Brier"] = [round(brier_score_loss(y_test, p), 3) for p in probs.values()]
    return tbl, probs


# ----------------------------------------------------------------- subgroups
def subgroup_table(ds, oof_all):
    y_np, site_np = np.asarray(ds.y), np.asarray(ds.site)

    def row(label, mask):
        yt = y_np[mask]
        pos, neg = int(yt.sum()), int(len(yt) - yt.sum())
        auc = roc_auc_score(yt, oof_all[mask]) if min(pos, neg) >= 10 else np.nan
        return {"subgroup": label, "n": int(mask.sum()), "disease rate": round(yt.mean(), 2),
                "AUC": round(auc, 3) if not np.isnan(auc) else "n/a (too few)"}

    age, sex = ds.X["age"].values, ds.X["sex"].values
    sp = ds.sparse.values
    rows = [row("all patients", np.ones(len(y_np), bool)), row("female", sex == 0), row("male", sex == 1),
            row("age < 50", age < 50), row("age 50-59", (age >= 50) & (age < 60)), row("age 60+", age >= 60)]
    rows += [row(f"hospital: {s}", site_np == s) for s in SITES]
    rows += [row("sparse records (3+ core values missing)", sp), row("dense records", ~sp)]
    return pd.DataFrame(rows)


# ------------------------------------------- leave-one-hospital-out + recalibration
def leave_one_hospital_out(ds, final_cols, settings):
    """Train on three hospitals (re-tuning each time), test on the fourth; with and without flags."""
    rows, preds = [], {}
    for s in SITES:
        mask = (ds.site == s).values
        share = ds.y[mask].mean()
        if min(share, 1 - share) < 0.10:
            print(f"  {s:12s} skipped (only {min(share, 1 - share):.0%} minority class, AUC meaningless)")
            continue
        for flags in (False, True):
            for name in MODEL_NAMES:
                rs = make_search(name, final_cols, settings, n_jobs=-1, flags=flags)
                rs.fit(ds.X[~mask], ds.y[~mask])
                p = rs.predict_proba(ds.X[mask])[:, 1]
                if not flags:
                    preds[(name, s)] = (ds.y[mask].values, p, np.where(mask)[0])
                rows.append({"hospital": s, "n": int(mask.sum()), "model": name,
                             "missing-flags": "yes" if flags else "no",
                             "AUC": round(roc_auc_score(ds.y[mask], p), 3),
                             "Brier": round(brier_score_loss(ds.y[mask], p), 3),
                             "actual_prev": round(share, 2), "mean_pred": round(p.mean(), 2)})
    return pd.DataFrame(rows), preds


def _logit(q):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def recalibration_demo(loso_preds, best_name, settings):
    """Use 30% of a new hospital's patients to re-fit intercept+slope; evaluate on the other 70%."""
    rows, gains = [], []
    for s in SITES:
        if (best_name, s) not in loso_preds:
            continue
        yt, p, _ = loso_preds[(best_name, s)]
        before, after = [], []
        splitter = StratifiedShuffleSplit(settings.recal_reps, test_size=0.7, random_state=settings.seed)
        for cal, ev in splitter.split(p, yt):
            lr = LogisticRegression(C=1e6).fit(_logit(p[cal]).reshape(-1, 1), yt[cal])
            p2 = lr.predict_proba(_logit(p[ev]).reshape(-1, 1))[:, 1]
            before.append(brier_score_loss(yt[ev], p[ev]))
            after.append(brier_score_loss(yt[ev], p2))
        gains.append(1 - np.mean(after) / np.mean(before))
        rows.append({"hospital": s, "Brier before": round(np.mean(before), 3),
                     "Brier after": round(np.mean(after), 3), "improvement": f"{gains[-1]:.0%}"})
    return pd.DataFrame(rows), gains
