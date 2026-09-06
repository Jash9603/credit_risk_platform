"""Per-prediction explanations using SHAP's TreeExplainer.

Turns "the model says 73% risk" into a short, ranked list of which features pushed that
particular applicant's score up or down, in plain English rather than raw SHAP numbers —
what a non-technical credit officer actually needs to see.

Run: `python -m src.explainability.shap_explainer --sk-id 100002`
"""
import argparse

import shap

from src.ml.predict import build_model_input, get_applicant_row, load_artifacts
from src.utils.feature_labels import format_value, readable_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_N = 5


def explain(sk_id_curr: int, top_n: int = TOP_N) -> list:
    model, metadata = load_artifacts()
    applicant = get_applicant_row(sk_id_curr)
    X = build_model_input(applicant, metadata)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Some shap/LightGBM version combinations return a list (one array per class) instead
    # of a single 2D array — normalise to the positive ("will default") class either way.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_values = shap_values[0]
    ranked = sorted(
        zip(X.columns, row_values, X.iloc[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_n]

    explanations = []
    for feature, shap_value, actual_value in ranked:
        direction = "increased" if shap_value > 0 else "decreased"
        explanations.append({
            "feature": feature,
            "value": format_value(actual_value),
            "shap_value": round(float(shap_value), 4),
            "sentence": f"{readable_name(feature)} ({format_value(actual_value)}) {direction} the risk score",
        })
    return explanations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sk-id", type=int, required=True)
    args = parser.parse_args()

    for item in explain(args.sk_id):
        print(item["sentence"])
