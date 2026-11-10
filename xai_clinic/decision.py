"""Decision analysis: thresholds, operating points, net benefit and risk tiers."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import TIER_CUTS


def risk_tier(p, cuts=TIER_CUTS):
    p = np.asarray(p, dtype=float)
    return np.where(p < cuts[0], "Low", np.where(p < cuts[1], "Moderate", "High"))


def choose_thresholds(oof, y_train, settings):
    """Pick thresholds on TRAIN out-of-fold predictions only.

    cost-optimal minimises   cost_fn * prev * (1 - sensitivity) + (1 - prev) * false-positive-rate
    at the DEPLOYMENT prevalence (not the ~55% of the training data).
    """
    oof, y = np.asarray(oof), np.asarray(y_train)
    fpr, tpr, thr = roc_curve(y, oof)
    thr_sens = float(thr[np.argmax(tpr >= 0.90)])               # first threshold reaching 90% recall
    grid = np.linspace(0.05, 0.95, 91)
    sens = np.array([((oof >= t) & (y == 1)).sum() / (y == 1).sum() for t in grid])
    fpr_t = np.array([((oof >= t) & (y == 0)).sum() / (y == 0).sum() for t in grid])
    prev = settings.deploy_prevalence
    cost = settings.cost_fn_to_fp * prev * (1 - sens) + (1 - prev) * fpr_t
    return {"default (0.50)": 0.5, "90%-sensitivity": thr_sens, "cost-optimal": float(grid[int(np.argmin(cost))])}


def operating_points(y_test, p, thresholds, settings):
    """Impact per 1,000 screened patients at the assumed deployment prevalence."""
    yt, prev = np.asarray(y_test), settings.deploy_prevalence
    rows = []
    for name, t in thresholds.items():
        pred = np.asarray(p) >= t
        sens = ((pred == 1) & (yt == 1)).sum() / (yt == 1).sum()
        spec = ((pred == 0) & (yt == 0)).sum() / (yt == 0).sum()
        tp, fn = 1000 * prev * sens, 1000 * prev * (1 - sens)
        fp, tn = 1000 * (1 - prev) * (1 - spec), 1000 * (1 - prev) * spec
        rows.append({"operating point": name, "threshold": round(t, 2),
                     "sensitivity": round(sens, 3), "specificity": round(spec, 3),
                     "flagged /1000": round(tp + fp), "cases caught": round(tp, 1),
                     "cases missed": round(fn, 1), "false alarms": round(fp),
                     "PPV": round(tp / (tp + fp), 3), "NPV": round(tn / (tn + fn), 4),
                     "cost /1000": round(settings.cost_fn_to_fp * fn + fp)})
    return pd.DataFrame(rows)


def net_benefit(yt, p, pt):
    yt, p = np.asarray(yt), np.asarray(p)
    pred = p >= pt
    n = len(yt)
    return ((pred & (yt == 1)).sum() - (pred & (yt == 0)).sum() * pt / (1 - pt)) / n


def tier_table(oof_all, y):
    t = pd.DataFrame({"tier": risk_tier(oof_all), "disease": np.asarray(y), "pred": oof_all})
    out = t.groupby("tier").agg(patients=("disease", "size"), observed_rate=("disease", "mean"),
                                mean_predicted=("pred", "mean"))
    return out.reindex(["Low", "Moderate", "High"]).round(3)
