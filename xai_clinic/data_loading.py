"""Data loading: find or download the four UCI files and parse them into one raw DataFrame."""
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from .config import COLS, DATA_DIR, SITES, ZIP_URL


def find_file(names):
    """Return the first existing path among the candidate names (data/ or current folder)."""
    for n in names:
        for folder in (DATA_DIR, "."):
            p = os.path.join(folder, n)
            if os.path.exists(p):
                return p
    return None


def ensure_data():
    """Download the UCI zip the first time, if any of the four files is missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if all(find_file(n) for n in SITES.values()):
        return
    print("Downloading UCI Heart Disease data ...")
    try:
        zip_path = os.path.join(DATA_DIR, "heart_disease.zip")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                base = os.path.basename(member)
                if base.startswith("processed.") and base.endswith(".data"):
                    with z.open(member) as src, open(os.path.join(DATA_DIR, base), "wb") as dst:
                        dst.write(src.read())
    except Exception as e:
        raise SystemExit(
            f"\nAuto-download failed ({e}).\nDownload the zip manually from\n  {ZIP_URL}\n"
            f"and put these 4 files inside the '{DATA_DIR}' folder:\n"
            "  processed.cleveland.data, processed.hungarian.data,\n"
            "  processed.switzerland.data, processed.va.data")


def load_data():
    """All four hospitals in one DataFrame with a `site` and a binary `target` column.

    Cleaning done here is *format* cleaning only: '?' -> NaN, and impossible zero
    cholesterol / blood pressure -> NaN. Statistical choices live in data_processing.
    """
    ensure_data()
    frames = []
    for site, names in SITES.items():
        d = pd.read_csv(find_file(names), header=None, names=COLS,
                        na_values="?", skipinitialspace=True)
        d["site"] = site
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["target"] = (df["num"] > 0).astype(int)       # 0 = no disease, 1 = disease found
    df = df.drop(columns="num")
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    return df
