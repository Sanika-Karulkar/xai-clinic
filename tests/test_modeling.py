import numpy as np
import pandas as pd

from xai_clinic.config import MODEL_NAMES, Settings
from xai_clinic.modeling import (GRIDS, boot_metrics, choose_best, make_search, paired_tests)


def test_every_model_has_a_search_grid_and_builds_a_search():
    s = Settings(quick=True)
    for name in MODEL_NAMES:
        assert name in GRIDS
        search = make_search(name, ["age", "sex", "oldpeak"], s)
        assert search.n_iter <= s.n_iter


def test_quick_settings_shrink_the_run():
    q, f = Settings(quick=True), Settings()
    assert q.n_boot < f.n_boot and q.n_iter < f.n_iter and q.rf_trees < f.rf_trees


def test_boot_metrics_point_estimate_and_ci_format():
    y = np.array([1] * 40 + [0] * 40)
    p = np.r_[np.full(40, 0.9), np.full(40, 0.1)]
    m = boot_metrics(y, p, 0.5, n_boot=100, seed=1)
    assert m["AUC"].startswith("1.000") and m["Accuracy"].startswith("1.000")
    assert set(m) == {"AUC", "Accuracy", "Sensitivity", "Specificity"}


def test_paired_tests_and_best_model_choice():
    r = np.random.default_rng(0)
    base = 0.88 + r.normal(0, 0.01, 10)
    nested = {"Logistic Regression": base + r.normal(0, 0.006, 10),
              "Random Forest": base + 0.002 + r.normal(0, 0.006, 10),
              "Gradient Boosting": base - 0.001 + r.normal(0, 0.006, 10)}
    pairs = paired_tests(nested)
    assert len(pairs) == 3 and (pairs["p-value"] > 0.05).all()
    best, tied = choose_best(nested, pairs, Settings())
    assert best == max(nested, key=lambda n: nested[n].mean()) and tied
    forced, _ = choose_best(nested, pairs, Settings(force_model="Logistic Regression"))
    assert forced == "Logistic Regression"


def test_paired_tests_handle_identical_scores_without_dividing_by_zero():
    same = np.full(10, 0.88)
    pairs = paired_tests({"Logistic Regression": same, "Random Forest": same.copy(),
                          "Gradient Boosting": same.copy()})
    assert (pairs["p-value"] == 1.0).all()
