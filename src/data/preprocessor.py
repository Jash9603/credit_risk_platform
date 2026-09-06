"""Clean the raw application data and add a small set of engineered features.

Kept deliberately simple: fix the known data issues, add a handful of ratios that are
easy to explain to a non-technical reader, and pull in a couple of summary numbers from
the applicant's credit history. Everything here is one function per step so the same
code path works for a full training set or a single new applicant at prediction time —
no need for a heavier scikit-learn Pipeline object.

Numeric columns are deliberately left with real NaNs where they occur (LightGBM handles
missing values natively and often makes good use of "this was missing" as a signal —
filling it in with a mean/median would actually throw that information away). Categorical
columns are filled with "Unknown" instead, since a tree model needs a concrete category
to split on.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import load_application_train, load_table
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DAYS_EMPLOYED_ANOMALY = 365243

# DAYS_* columns are stored as negative day-offsets, which is awkward to read. Converting
# to positive years makes charts, SHAP explanations, and business rules much easier to
# follow later — "age 34" beats "DAYS_BIRTH = -12490" every time.
DAYS_TO_YEARS_COLUMNS = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "YEARS_EMPLOYED",
    "DAYS_REGISTRATION": "YEARS_REGISTRATION",
    "DAYS_ID_PUBLISH": "YEARS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE": "YEARS_LAST_PHONE_CHANGE",
}


def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Two known data issues in this dataset (found during EDA): a sentinel value used
    for "no current job", and a handful of rows with an invalid gender code."""
    df = df.copy()

    df["IS_DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    return df


def convert_days_to_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for days_col, years_col in DAYS_TO_YEARS_COLUMNS.items():
        df[years_col] = -df[days_col] / 365.25
        df = df.drop(columns=[days_col])
    return df


def engineer_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """A handful of simple, explainable ratios — the kind of number a credit officer
    would recognise, not a black-box combination."""
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"]
    return df


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant, summarising their history with *other* lenders (reported
    via the credit bureau) — see the EDA notebook, insight #6."""
    agg = bureau.groupby("SK_ID_CURR").agg(
        N_BUREAU_LOANS=("SK_ID_BUREAU", "count"),
        N_BUREAU_ACTIVE=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        N_BUREAU_BAD_DEBT=("CREDIT_ACTIVE", lambda s: (s == "Bad debt").sum()),
        AVG_BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "mean"),
    ).reset_index()
    return agg


def previous_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant, summarising their own history of applications with Home
    Credit itself (separate from the bureau's external view)."""
    agg = prev.groupby("SK_ID_CURR").agg(
        N_PREV_APPLICATIONS=("SK_ID_PREV", "count"),
        N_PREV_REFUSED=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").sum()),
        N_PREV_APPROVED=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").sum()),
    ).reset_index()
    agg["PREV_REFUSED_RATIO"] = agg["N_PREV_REFUSED"] / agg["N_PREV_APPLICATIONS"]
    return agg


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing text values with an explicit "Unknown" category and mark every
    non-numeric column as `category` dtype, which LightGBM can split on directly."""
    df = df.copy()
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype("category")
    return df


def build_feature_matrix(app_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run the full cleaning + feature-engineering sequence. Pass a single-row DataFrame
    at prediction time to run the identical steps on one new applicant; leave it blank to
    build the full training matrix."""
    df = load_application_train() if app_df is None else app_df.copy()

    df = fix_known_anomalies(df)
    df = convert_days_to_years(df)
    df = engineer_ratios(df)

    bureau = load_table("bureau")
    df = df.merge(bureau_features(bureau), on="SK_ID_CURR", how="left")
    bureau_count_cols = ["N_BUREAU_LOANS", "N_BUREAU_ACTIVE", "N_BUREAU_BAD_DEBT"]
    df[bureau_count_cols] = df[bureau_count_cols].fillna(0)
    # AVG_BUREAU_CREDIT_SUM is left as NaN when there's no bureau history — "no data" is
    # not the same as "zero average credit", so it shouldn't be filled with 0.

    prev = load_table("previous_application")
    df = df.merge(previous_application_features(prev), on="SK_ID_CURR", how="left")
    prev_count_cols = ["N_PREV_APPLICATIONS", "N_PREV_REFUSED", "N_PREV_APPROVED"]
    df[prev_count_cols] = df[prev_count_cols].fillna(0)
    # PREV_REFUSED_RATIO stays NaN with no prior applications, same reasoning as above.

    df = prepare_categoricals(df)
    return df


if __name__ == "__main__":
    matrix = build_feature_matrix()
    print(f"Feature matrix: {matrix.shape[0]:,} rows, {matrix.shape[1]} columns")

    cat_cols = matrix.select_dtypes(include="category").columns
    cat_missing = int(matrix[cat_cols].isna().sum().sum())
    print(f"Missing values in categorical columns (should be 0): {cat_missing}")

    numeric = matrix.select_dtypes(exclude="category")
    numeric_missing = numeric.isna().sum()
    numeric_missing = numeric_missing[numeric_missing > 0]
    print(f"Numeric columns with missing values (expected — left for the model): {len(numeric_missing)}")
    print(numeric_missing.sort_values(ascending=False).head(10))
