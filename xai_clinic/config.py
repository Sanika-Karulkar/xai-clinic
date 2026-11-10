"""Constants and run settings. Nothing in here does any work."""
from dataclasses import dataclass

DATA_DIR = "data"
FIG_DIR = "figures"
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

SITES = {                                   # hospital -> accepted file names
    "cleveland":   ["processed.cleveland.data", "processed_cleveland.data"],
    "hungarian":   ["processed.hungarian.data"],
    "switzerland": ["processed.switzerland.data"],
    "va":          ["processed.va.data"],
}
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
RAW_FEATURES = [c for c in COLS if c != "num"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]            # codes, not quantities
ENGINEERED_ALL = ["hr_reserve", "chol_per_age", "bp_chol_index", "high_risk_age"]
ENGINEERED_FINAL = ["hr_reserve", "chol_per_age"]           # the two that are not trivial re-labels
CORE_MEASUREMENTS = ["trestbps", "thalach", "exang", "oldpeak"]   # 3+ missing = "sparse record"
MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
TIER_CUTS = (0.30, 0.70)                                    # Low < 0.30 <= Moderate < 0.70 <= High

PRETTY = {"cp": "chest pain type", "exang": "exercise-induced angina", "oldpeak": "ST depression",
          "thalach": "max heart rate", "slope": "ST slope", "thal": "thalassemia test",
          "sex": "sex", "age": "age", "trestbps": "resting blood pressure",
          "chol": "cholesterol", "fbs": "fasting blood sugar > 120", "restecg": "resting ECG",
          "hr_reserve": "heart-rate reserve", "chol_per_age": "cholesterol / age",
          "bp_chol_index": "BP x cholesterol", "high_risk_age": "age-sex risk flag"}


@dataclass
class Settings:
    """Everything you may want to change lives here (or on the command line)."""
    seed: int = 42
    missing_threshold: float = 0.60     # drop features missing in more than 60% of patients
    primary_flags: bool = False         # PRIMARY model is hospital-agnostic; flags = sensitivity run
    n_boot: int = 2000                  # bootstrap resamples for confidence intervals
    n_iter: int = 15                    # random-search configurations per model
    nested_repeats: int = 2
    ablation_repeats: int = 3
    recal_reps: int = 20
    rf_trees: int = 300
    # business assumptions (state them in your README)
    cost_fn_to_fp: float = 5.0          # a missed patient costs this many false alarms
    deploy_prevalence: float = 0.10     # disease rate of the population you would screen
    force_model: str | None = None      # e.g. "Logistic Regression"; None = best nested-CV mean
    quick: bool = False                 # smoke-test mode: tiny search, rough numbers

    def __post_init__(self):
        if self.quick:
            self.n_boot, self.n_iter, self.nested_repeats = 200, 3, 1
            self.ablation_repeats, self.recal_reps, self.rf_trees = 1, 5, 50
