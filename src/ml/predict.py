"""Score a single loan applicant: default probability, a 0-1000 risk score, and a
Low/Medium/High risk band.

Run: `python -m src.ml.predict --sk-id 100002`
"""
import argparse
from functools import lru_cache

import joblib
import pandas as pd

from src.data.loader import load_application_train
from src.data.preprocessor import apply_categories, build_feature_matrix
from src.ml.evaluate import risk_band
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_artifacts():
    """Model + everything needed to use it correctly, in one file — feature order,
    which columns are categorical (and their exact training-time categories), and the
    risk-band cutoffs. Cached so a long-running process (the API server) deserialises the
    ~1.3MB model file once, not on every request."""
    bundle = joblib.load(settings.models_path / "model.pkl")
    model = bundle["model"]
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    return model, metadata


def get_applicant_row(sk_id_curr: int) -> pd.DataFrame:
    app = load_application_train()
    row = app[app["SK_ID_CURR"] == sk_id_curr]
    if row.empty:
        raise ValueError(f"SK_ID_CURR {sk_id_curr} not found in application_train.")
    return row


def build_model_input(applicant: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Run the same cleaning/feature-engineering as training, then force categorical
    columns to match the exact category set the model was trained on."""
    features = build_feature_matrix(app_df=applicant)
    features = apply_categories(features, metadata["categorical_categories"])
    return features[metadata["feature_names"]]


def predict(sk_id_curr: int) -> dict:
    model, metadata = load_artifacts()
    applicant = get_applicant_row(sk_id_curr)
    X = build_model_input(applicant, metadata)

    probability = float(model.predict_proba(X)[:, 1][0])
    band = risk_band(probability, metadata["risk_band_cutoffs"])

    return {
        "sk_id_curr": sk_id_curr,
        "default_probability": round(probability, 4),
        "risk_score": round(probability * 1000),
        "risk_band": band,
        "actual_target": int(applicant["TARGET"].iloc[0]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sk-id", type=int, required=True)
    args = parser.parse_args()

    for key, value in predict(args.sk_id).items():
        print(f"{key}: {value}")
