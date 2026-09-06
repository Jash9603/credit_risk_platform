"""Metrics and risk-banding logic for the default-risk model.

Kept separate from train.py so predict.py / the API can reuse the same threshold and
risk-band logic later without importing the training code.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Bottom 60% of predicted probability = Low, next 30% = Medium, top 10% = High.
# Percentile-based, not fixed values — see DECISIONS.md: scale_pos_weight (needed for the
# class imbalance) pushes raw probabilities well above their "textbook" range, so a fixed
# cutoff like 0.20 would land almost everyone in "High". Percentiles stay meaningful
# regardless of how the raw scores are calibrated.
BAND_PERCENTILES = {"low": 60, "medium": 90}


def compute_band_cutoffs(y_prob) -> dict:
    return {
        "low_cutoff": float(np.percentile(y_prob, BAND_PERCENTILES["low"])),
        "medium_cutoff": float(np.percentile(y_prob, BAND_PERCENTILES["medium"])),
    }


def risk_band(probability: float, cutoffs: dict) -> str:
    if probability < cutoffs["low_cutoff"]:
        return "Low"
    if probability < cutoffs["medium_cutoff"]:
        return "Medium"
    return "High"


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov statistic — the biggest gap between the true-positive rate and
    false-positive rate across all thresholds. A standard credit-risk metric alongside
    AUC: it asks "how well does the score separate the two groups", not just "how well is
    it ranked overall"."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true, y_prob) -> float:
    """Pick the probability cutoff that maximises F1, searched across the model's actual
    output range rather than assumed to be near 0.5 — scale_pos_weight shifts predicted
    probabilities upward, so the useful operating point can sit well above 0.5."""
    thresholds = np.linspace(0.01, 0.95, 95)
    scores = [f1_score(y_true, y_prob >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_predictions(y_true, y_prob) -> dict:
    threshold = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    band_cutoffs = compute_band_cutoffs(y_prob)
    bands = [risk_band(p, band_cutoffs) for p in y_prob]
    band_counts = {label: bands.count(label) for label in ("Low", "Medium", "High")}

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "ks_statistic": round(ks_statistic(y_true, y_prob), 4),
        "chosen_threshold": round(threshold, 3),
        "precision_at_threshold": round(float(precision_score(y_true, y_pred)), 4),
        "recall_at_threshold": round(float(recall_score(y_true, y_pred)), 4),
        "f1_at_threshold": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "risk_band_cutoffs": band_cutoffs,
        "risk_band_distribution": band_counts,
    }
