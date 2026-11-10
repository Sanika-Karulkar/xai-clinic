"""Data processing: de-duplication, missing-data handling, feature engineering,
train/test split, and the leak-free preprocessing pipeline."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (CATEGORICAL, CORE_MEASUREMENTS, ENGINEERED_ALL, ENGINEERED_FINAL,
                     PRETTY, RAW_FEATURES)


# ----------------------------------------------------------------- cleaning
def deduplicate(df):
    n = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), n


def missingness(df):
    """% missing per feature overall, and per hospital (missingness is a hospital fingerprint)."""
    overall = (df[RAW_FEATURES].isna().mean() * 100).sort_values(ascending=False)
    by_site = df.groupby("site")[RAW_FEATURES].apply(lambda g: g.isna().mean() * 100).round(0)
    return overall, by_site


def drop_sparse_features(df, threshold):
    dropped = [c for c in RAW_FEATURES if df[c].isna().mean() > threshold]
    return df.drop(columns=dropped), dropped


def sparse_records(df):
    """True for patients with 3+ of the 4 core measurements missing (mostly VA)."""
    return df[CORE_MEASUREMENTS].isna().sum(axis=1) >= 3


# ------------------------------------------------------ feature engineering
def engineer(d):
    d = d.copy()
    d["hr_reserve"] = (220 - d["age"]) - d["thalach"]          # predicted max HR - achieved
    d["chol_per_age"] = d["chol"] / d["age"]
    d["bp_chol_index"] = d["trestbps"] * d["chol"] / 1000
    d["high_risk_age"] = (((d["sex"] == 1) & (d["age"] >= 45)) |
                          ((d["sex"] == 0) & (d["age"] >= 55))).astype(int)
    for c in CATEGORICAL:                                      # keep NaN as NaN (imputed inside the pipeline)
        if c in d.columns:
            d[c] = d[c].apply(lambda v: np.nan if pd.isna(v) else str(int(v)))
    return d


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    site: pd.Series
    sparse: pd.Series
    raw_cols: list
    final_cols: list


def build_dataset(df):
    """df (after de-dup and dropping very-sparse features) -> model-ready Dataset."""
    sparse = sparse_records(df)
    data = engineer(df)
    X = data.drop(columns=["target", "site"])
    raw_cols = [c for c in df.columns if c not in ("target", "site")]
    return Dataset(X=X, y=data["target"], site=data["site"], sparse=sparse,
                   raw_cols=raw_cols, final_cols=raw_cols + ENGINEERED_FINAL)


def split_dataset(ds, seed, test_size=0.2):
    """Stratified by disease AND hospital so every hospital appears in both parts."""
    strata = ds.y.astype(str) + "_" + ds.site
    return train_test_split(ds.X, ds.y, test_size=test_size, stratify=strata, random_state=seed)


# ----------------------------------------------------------------- pipeline
def make_pipeline(model, cols, scale=False, flags=False):
    """Leak-free pipeline: imputation lives INSIDE so it is re-fit on every training fold.

    flags=False (PRIMARY): plain median / most-frequent imputation. The model cannot see
                           *which tests were recorded*, so it cannot identify the hospital.
    flags=True  (sensitivity analysis): missing-indicators on RAW numeric columns only,
                           plus a 'missing' category for categorical columns.
    """
    cat = [c for c in cols if c in CATEGORICAL]
    eng = [c for c in cols if c in ENGINEERED_ALL]
    raw_num = [c for c in cols if c not in CATEGORICAL and c not in ENGINEERED_ALL]
    cat_imp = (SimpleImputer(strategy="constant", fill_value="missing") if flags
               else SimpleImputer(strategy="most_frequent"))
    parts = [("num", SimpleImputer(strategy="median", add_indicator=flags), raw_num)]
    if eng:                                    # engineered columns never get their own flags
        parts.append(("eng", SimpleImputer(strategy="median"), eng))
    if cat:
        parts.append(("cat", Pipeline([("imp", cat_imp),
                                       ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                             sparse_output=False))]), cat))
    pre = ColumnTransformer(parts, verbose_feature_names_out=False).set_output(transform="pandas")
    steps = [("pre", pre)]
    if scale:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    steps.append(("model", model))
    return Pipeline(steps)


# ------------------------------------------------------------ feature names
def group_name(feature):
    """Map a one-hot / indicator column back to the clinical variable it came from."""
    if feature.startswith("missingindicator_"):
        return "missing flag: " + feature[len("missingindicator_"):]
    for c in CATEGORICAL:
        if feature.startswith(c + "_"):
            return c
    return feature


def pretty(name):
    return PRETTY.get(name, name)
