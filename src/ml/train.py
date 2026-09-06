"""Train the default-risk model.

Two models get trained, not just one:
- Logistic Regression — a fast, fully interpretable baseline. Needs imputed/encoded/
  scaled input since it can't use raw NaN values or category columns directly.
- LightGBM — the model actually shipped. Handles missing values and categorical columns
  natively, and is expected to clearly beat the baseline on this kind of mixed-type
  tabular data.

Class imbalance (~8% positive) is handled with LightGBM's scale_pos_weight rather than
resampling (SMOTE, undersampling) — it reweights the real data instead of synthesising or
discarding rows, which is simpler to reason about and reproduce.

Run: `python -m src.ml.train`
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessor import build_feature_matrix, get_categories
from src.ml.evaluate import compute_band_cutoffs, evaluate_predictions
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_FOLDS = 5
RANDOM_STATE = 42

# Fixed, reasonable defaults rather than a tuned search — see DECISIONS.md.
LGBM_PARAMS = {
    "objective": "binary",
    # LightGBM auto-tracks the objective's own metric (binary_logloss) alongside whatever
    # eval_metric is passed to fit(), and uses it for the early-stopping decision. Under
    # scale_pos_weight the raw probabilities are deliberately skewed, so logloss looks
    # like it's getting worse while AUC (rank-based, scale-insensitive) keeps improving —
    # metric="None" stops that auto-tracked logloss from hijacking early stopping.
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 100


def train_baseline(X: pd.DataFrame, y: pd.Series) -> float:
    """Logistic Regression on a single train/valid split — just enough to give the
    LightGBM comparison a real number, not meant to be tuned or shipped."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    numeric_cols = X.select_dtypes(exclude="category").columns.tolist()
    categorical_cols = X.select_dtypes(include="category").columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    baseline = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)
    valid_prob = baseline.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, valid_prob)


def train_lightgbm_cv(X: pd.DataFrame, y: pd.Series, categorical_cols: list):
    """Stratified K-fold CV. Returns out-of-fold predictions (used for the real evaluation
    metrics, since scoring a model on data it trained on would be optimistic) and the
    average best iteration (used to size the final full-data model)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(X))
    best_iterations = []
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(
            **LGBM_PARAMS, n_estimators=N_ESTIMATORS, scale_pos_weight=scale_pos_weight
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof_pred[valid_idx] = model.predict_proba(X_valid)[:, 1]
        best_iterations.append(model.best_iteration_)
        fold_auc = roc_auc_score(y_valid, oof_pred[valid_idx])
        logger.info("Fold %d/%d — AUC %.4f, best_iteration %d", fold, N_FOLDS, fold_auc, model.best_iteration_)

    return oof_pred, int(np.mean(best_iterations)), scale_pos_weight


def train_final_model(X, y, categorical_cols, n_estimators, scale_pos_weight):
    model = lgb.LGBMClassifier(
        **LGBM_PARAMS, n_estimators=n_estimators, scale_pos_weight=scale_pos_weight
    )
    model.fit(X, y, categorical_feature=categorical_cols)
    return model


def main():
    matrix = build_feature_matrix()
    X = matrix.drop(columns=["SK_ID_CURR", "TARGET"])
    y = matrix["TARGET"]
    categorical_cols = X.select_dtypes(include="category").columns.tolist()

    logger.info("Training Logistic Regression baseline...")
    baseline_auc = train_baseline(X, y)
    logger.info("Baseline (Logistic Regression) ROC-AUC: %.4f", baseline_auc)

    logger.info("Running %d-fold cross-validation for LightGBM...", N_FOLDS)
    oof_pred, final_n_estimators, scale_pos_weight = train_lightgbm_cv(X, y, categorical_cols)

    metrics = evaluate_predictions(y, oof_pred)
    metrics["baseline_logistic_regression_auc"] = round(float(baseline_auc), 4)
    metrics["lightgbm_cv_auc"] = round(float(roc_auc_score(y, oof_pred)), 4)
    metrics["scale_pos_weight_used"] = round(float(scale_pos_weight), 2)
    metrics["final_model_n_estimators"] = final_n_estimators

    logger.info("Training final model on the full training set...")
    final_model = train_final_model(X, y, categorical_cols, final_n_estimators, scale_pos_weight)

    settings.models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": X.columns.tolist(),
            "categorical_columns": categorical_cols,
            "categorical_categories": get_categories(X),
            "risk_band_cutoffs": compute_band_cutoffs(oof_pred),
        },
        settings.models_path / "model.pkl",
    )

    settings.artifacts_path.mkdir(parents=True, exist_ok=True)
    with open(settings.artifacts_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    logger.info("Saved model to %s", settings.models_path / "model.pkl")


if __name__ == "__main__":
    main()
