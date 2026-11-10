"""
XAI Clinic - interactive demo: enter a patient's measurements, get a risk score, a risk tier and a
per-patient SHAP explanation.     Run:  uv run streamlit run app.py
(first run `uv run xai-clinic` so figures/model_artifact.joblib exists)

EDUCATIONAL DEMO ONLY - not a medical device.
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from xai_clinic.data_processing import engineer
from xai_clinic.decision import risk_tier
from xai_clinic.explain import explain

ARTIFACT = os.path.join("figures", "model_artifact.joblib")
st.set_page_config(page_title="XAI Clinic", page_icon="🫀", layout="wide")


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT)


st.title("XAI Clinic: explainable heart-disease risk demo")
st.warning("Educational demo trained on 1980s data from four hospitals. "
           "**Not a medical device - do not use for health decisions.**")

if not os.path.exists(ARTIFACT):
    st.error("Model file not found. Run `uv run xai-clinic` first to create figures/model_artifact.joblib.")
    st.stop()
art = load_artifact()
UNKNOWN = "Unknown / not measured"

with st.sidebar:
    st.header("Patient measurements")
    age = st.slider("Age (years)", 25, 80, 55)
    sex = st.selectbox("Sex", {0: "Female", 1: "Male"}.items(), format_func=lambda kv: kv[1])[0]
    cp = st.selectbox("Chest pain type", {1: "Typical angina", 2: "Atypical angina",
                                          3: "Non-anginal pain", 4: "Asymptomatic"}.items(),
                      format_func=lambda kv: kv[1], index=3)[0]
    trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", {0: "No", 1: "Yes"}.items(),
                       format_func=lambda kv: kv[1])[0]
    restecg = st.selectbox("Resting ECG", {0: "Normal", 1: "ST-T abnormality",
                                           2: "LV hypertrophy"}.items(), format_func=lambda kv: kv[1])[0]
    thalach = st.slider("Max heart rate achieved", 60, 210, 150)
    exang = st.selectbox("Exercise-induced angina", {0: "No", 1: "Yes"}.items(),
                         format_func=lambda kv: kv[1])[0]
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox("ST slope", [UNKNOWN, 1, 2, 3],
                         format_func=lambda v: v if v == UNKNOWN else {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[v])
    thal = st.selectbox("Thalassemia test", [UNKNOWN, 3, 6, 7],
                        format_func=lambda v: v if v == UNKNOWN else {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[v])

row = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
       "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
       "slope": np.nan if slope == UNKNOWN else slope, "thal": np.nan if thal == UNKNOWN else thal}
X_row = engineer(pd.DataFrame([row])).reindex(columns=art["input_columns"])
p = float(art["pipeline"].predict_proba(X_row)[0, 1])
tier = str(risk_tier(p))
thr = art["thresholds"]

c1, c2, c3 = st.columns(3)
c1.metric("Model risk score", f"{p:.0%}")
c2.metric("Risk tier", tier)
c3.metric("Model", art["model_name"])
st.progress(min(max(p, 0.0), 1.0))
flags = {name: ("above" if p >= t else "below") for name, t in thr.items()}
st.caption("Versus decision thresholds: " + "; ".join(f"{k} ({thr[k]:.2f}): **{v}**" for k, v in flags.items()))
st.info("The score is a model output, not a calibrated probability for any particular hospital: "
        "the analysis showed risk estimates drift when the model is moved to a new site.")

st.subheader("Why this score? (SHAP, per-patient)")
sv, _ = explain(art["pipeline"], art["background"], X_row)
fig = plt.figure()
shap.plots.waterfall(sv[0], max_display=10, show=False)
st.pyplot(plt.gcf(), clear_figure=True)
st.caption("Red bars push the score up, blue bars push it down. Units are log-odds for logistic regression "
           "and probability for tree models.")

with st.expander("Model details and evaluation"):
    m = art["metrics"]
    st.write(f"Built {m['built']} on {m['n_patients']} patients. Nested-CV AUC {m['nested_cv_auc']:.3f}.")
    st.write("Held-out test performance (95% bootstrap CI):")
    st.json(m["test"])
    st.write("See figures/MODEL_CARD.md for subgroup, cross-hospital and calibration results.")
