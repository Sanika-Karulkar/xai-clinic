"""Explainability: SHAP values, grouping dummies back to clinical variables, model agreement."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import MODEL_NAMES
from .data_processing import group_name, pretty


def explain(pipe, X_ref, X_eval):
    """SHAP values (for the 'disease' class) plus the transformed features the model sees."""
    import shap
    pre, model = pipe[:-1], pipe[-1]
    A = pre.transform(X_eval)
    if isinstance(model, LogisticRegression):
        bg = pre.transform(X_ref)
        sv = shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)))(A)   # log-odds
    else:
        sv = shap.TreeExplainer(model)(A, check_additivity=False)
        if sv.values.ndim == 3:                                   # Random Forest -> disease class
            sv = sv[:, :, 1]
    return sv, A


def group_shap(sv, columns):
    """Add up SHAP values of one-hot dummies per original variable (additive, so exact)."""
    G = pd.DataFrame(sv.values, columns=columns)
    return G.T.groupby([group_name(c) for c in columns]).sum().T


def importance_by_model(tuned, X_train, X_test):
    """SHAP for all three models; importance as a share of total, grouped by clinical variable."""
    sv_all, share = {}, {}
    for name in MODEL_NAMES:
        sv_all[name], A = explain(tuned[name], X_train, X_test)
        imp = group_shap(sv_all[name], A.columns).abs().mean()
        share[name] = imp / imp.sum()
    imp_df = pd.DataFrame(share)
    imp_df.index = [pretty(i) for i in imp_df.index]
    return sv_all, imp_df


def agreement(imp_df):
    """Spearman rank correlation between models, and top-5 overlaps."""
    rho = imp_df.corr(method="spearman").round(2)
    top5 = {n: set(imp_df[n].nlargest(5).index) for n in MODEL_NAMES}
    overlaps = {f"{a} vs {b}": len(top5[a] & top5[b])
                for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i + 1:]}
    return rho, overlaps
