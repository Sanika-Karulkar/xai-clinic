import numpy as np
import pandas as pd

from xai_clinic.config import COLS, SITES
from xai_clinic.data_loading import find_file, load_data


def _write_site_files(tmp_path):
    (tmp_path / "data").mkdir()
    row_ok = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0"
    row_missing = "67.0,1.0,4.0,0.0,0.0,0.0,2.0,108.0,1.0,1.5,?,?,?,2"     # zeros + '?' + disease level 2
    for site, names in SITES.items():
        (tmp_path / "data" / names[0]).write_text(row_ok + "\n" + row_missing + "\n")


def test_load_data_parses_missing_zero_and_target(tmp_path, monkeypatch):
    _write_site_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    df = load_data()
    assert len(df) == 2 * len(SITES) and set(df["site"]) == set(SITES)
    assert "num" not in df.columns and set(df.columns) >= set(COLS) - {"num"} | {"target", "site"}
    bad = df[df["trestbps"].isna()]
    assert len(bad) == len(SITES)                      # zero blood pressure became NaN
    assert df["chol"].isna().sum() == len(SITES)       # zero cholesterol became NaN
    assert df["slope"].isna().sum() == len(SITES) and df["thal"].isna().sum() == len(SITES)   # '?' became NaN
    assert df["target"].tolist().count(1) == len(SITES) and df["target"].tolist().count(0) == len(SITES)


def test_find_file_checks_alternative_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "processed_cleveland.data").write_text("x")     # underscore variant
    assert find_file(SITES["cleveland"]).endswith("processed_cleveland.data")
    assert find_file(SITES["va"]) is None
