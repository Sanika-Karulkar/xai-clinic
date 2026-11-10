import numpy as np

from xai_clinic.config import Settings
from xai_clinic.decision import (choose_thresholds, net_benefit, operating_points, risk_tier,
                                 tier_table)


def scores(n=600, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    p = np.clip(0.35 + 0.3 * y + r.normal(0, 0.18, n), 0.01, 0.99)
    return y, p


def test_risk_tier_boundaries():
    assert list(risk_tier([0.0, 0.29, 0.30, 0.69, 0.70, 1.0])) == ["Low", "Low", "Moderate", "Moderate", "High", "High"]


def test_cost_threshold_falls_as_misses_get_costlier():
    y, p = scores()
    cheap = choose_thresholds(p, y, Settings(cost_fn_to_fp=1))["cost-optimal"]
    dear = choose_thresholds(p, y, Settings(cost_fn_to_fp=20))["cost-optimal"]
    assert dear <= cheap


def test_cost_threshold_uses_deployment_prevalence_not_training_prevalence():
    y, p = scores()                                              # training data is ~50% diseased
    common = choose_thresholds(p, y, Settings(deploy_prevalence=0.50))["cost-optimal"]
    rare = choose_thresholds(p, y, Settings(deploy_prevalence=0.02))["cost-optimal"]
    assert rare >= common                                       # rarer disease -> fewer flags -> higher threshold


def test_90_percent_sensitivity_threshold_really_reaches_90_percent():
    y, p = scores()
    t = choose_thresholds(p, y, Settings())["90%-sensitivity"]
    assert ((p >= t) & (y == 1)).sum() / (y == 1).sum() >= 0.90


def test_operating_points_counts_add_up_per_1000():
    y, p = scores()
    s = Settings(deploy_prevalence=0.10)
    op = operating_points(y, p, choose_thresholds(p, y, s), s)
    assert len(op) == 3
    for _, r in op.iterrows():
        assert abs(r["cases caught"] + r["cases missed"] - 100) < 0.2           # 10% of 1000
        assert abs(r["flagged /1000"] - (r["cases caught"] + r["false alarms"])) <= 1


def test_net_benefit_matches_hand_calculation():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.4, 0.6, 0.1])
    # threshold 0.5 flags patients 0 and 2: TP=1, FP=1, n=4 -> (1 - 1*0.5/0.5)/4 = 0
    assert net_benefit(y, p, 0.5) == 0.0


def test_tier_table_reports_observed_rate_per_tier():
    y = np.array([0, 0, 1, 1, 1, 0]); p = np.array([0.1, 0.2, 0.5, 0.55, 0.9, 0.95])
    t = tier_table(p, y)
    assert list(t.index) == ["Low", "Moderate", "High"] and t.loc["Low", "observed_rate"] == 0.0
    assert t["patients"].sum() == 6
