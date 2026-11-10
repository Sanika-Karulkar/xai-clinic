import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xai_clinic.config import ENGINEERED_ALL
from xai_clinic.data_processing import (build_dataset, deduplicate, drop_sparse_features, engineer,
                                        group_name, make_pipeline, missingness, split_dataset,
                                        sparse_records)


def fake_patients(n=150, seed=0, with_meta=False):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": r.integers(30, 77, n), "sex": r.integers(0, 2, n), "cp": r.integers(1, 5, n),
        "trestbps": r.integers(95, 200, n).astype(float), "chol": r.integers(130, 400, n).astype(float),
        "fbs": r.integers(0, 2, n), "restecg": r.integers(0, 3, n), "thalach": r.integers(80, 200, n),
        "exang": r.integers(0, 2, n), "oldpeak": r.random(n) * 5, "slope": r.integers(1, 4, n).astype(float),
        "thal": r.choice([3.0, 6.0, 7.0], n)})
    d.loc[r.random(n) < 0.1, "chol"] = np.nan
    d.loc[r.random(n) < 0.3, "thal"] = np.nan
    y = (d.oldpeak + 0.02 * d.age + d.exang + r.normal(0, 1, n) > 3.2).astype(int)
    if with_meta:
        d["ca"] = np.nan                                         # 100% missing -> must be dropped
        d["site"] = r.choice(["cleveland", "va"], n)
        d["target"] = y
    return d, y


def test_engineer_adds_features_and_keeps_missing_as_nan():
    d, _ = fake_patients()
    e = engineer(d)
    assert set(ENGINEERED_ALL) <= set(e.columns)
    assert e["thal"].isna().sum() == d["thal"].isna().sum()
    assert e["cp"].map(type).eq(str).all()


def test_hr_reserve_formula():
    d, _ = fake_patients(5)
    assert np.allclose(engineer(d)["hr_reserve"], (220 - d["age"]) - d["thalach"])


def test_pipeline_predicts_valid_probabilities_with_missing_values():
    d, y = fake_patients()
    X = engineer(d)
    cols = list(d.columns) + ["hr_reserve", "chol_per_age"]
    for flags in (False, True):
        pipe = make_pipeline(LogisticRegression(max_iter=1000), cols, scale=True, flags=flags).fit(X, y)
        p = pipe.predict_proba(X)[:, 1]
        assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1)) and not np.isnan(p).any()


def test_pipeline_handles_unseen_category_and_all_missing_row():
    d, y = fake_patients()
    X = engineer(d)
    pipe = make_pipeline(LogisticRegression(max_iter=1000), list(d.columns) + ["hr_reserve"], scale=True).fit(X, y)
    new = X.iloc[:1].copy()
    new["cp"] = "9"
    new[["chol", "thal", "slope"]] = np.nan
    assert 0 <= pipe.predict_proba(new)[0, 1] <= 1


def test_imputation_is_inside_the_pipeline_so_cv_cannot_leak():
    d, _ = fake_patients()
    assert make_pipeline(LogisticRegression(), list(d.columns)).steps[0][0] == "pre"


def test_flags_off_means_no_flag_columns_and_flags_only_on_raw_columns():
    d, y = fake_patients()
    X = engineer(d)
    cols = list(d.columns) + ["hr_reserve", "chol_per_age"]
    off = make_pipeline(LogisticRegression(max_iter=1000), cols, scale=True, flags=False).fit(X, y)
    names_off = list(off[:-1].transform(X).columns)
    assert not any(n.startswith("missingindicator") or n.endswith("_missing") for n in names_off)
    on = make_pipeline(LogisticRegression(max_iter=1000), cols, scale=True, flags=True).fit(X, y)
    names_on = list(on[:-1].transform(X).columns)
    assert "missingindicator_chol" in names_on
    assert "missingindicator_chol_per_age" not in names_on and "missingindicator_hr_reserve" not in names_on


def test_group_name_maps_dummies_back_to_variable():
    assert group_name("cp_4") == "cp" and group_name("thal_missing") == "thal"
    assert group_name("oldpeak") == "oldpeak"
    assert group_name("missingindicator_chol") == "missing flag: chol"


def test_deduplicate_and_missingness_and_sparse_feature_drop():
    d, _ = fake_patients(60, with_meta=True)
    d = pd.concat([d, d.iloc[:3]], ignore_index=True)           # 3 exact duplicates
    d2, n_dup = deduplicate(d)
    assert n_dup == 3 and len(d2) == 60
    overall, by_site = missingness(d2)
    assert overall["ca"] == 100 and set(by_site.index) == {"cleveland", "va"}
    d3, dropped = drop_sparse_features(d2, 0.60)
    assert dropped == ["ca"] and "ca" not in d3.columns


def test_sparse_records_flag():
    d, _ = fake_patients(5)
    d.loc[0, ["trestbps", "thalach", "exang"]] = np.nan          # 3 of 4 core values missing
    d.loc[1, ["trestbps", "thalach"]] = np.nan                   # only 2
    flag = sparse_records(d)
    assert bool(flag[0]) and not bool(flag[1])


def test_build_and_split_dataset_keeps_every_hospital_in_both_parts():
    d, _ = fake_patients(200, with_meta=True)
    d = d.drop(columns="ca")
    ds = build_dataset(d)
    assert "site" not in ds.X.columns and "target" not in ds.X.columns
    assert ds.final_cols[-2:] == ["hr_reserve", "chol_per_age"] and len(ds.sparse) == len(ds.X)
    Xtr, Xte, ytr, yte = split_dataset(ds, seed=1)
    assert len(Xte) == 40 and ds.site[Xtr.index].nunique() == 2 and ds.site[Xte.index].nunique() == 2
