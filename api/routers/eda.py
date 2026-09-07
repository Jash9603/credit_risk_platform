"""Serves the already-computed EDA results (from the notebook) as JSON + static chart
images — the API never re-runs analysis, it just exposes what's already in artifacts/eda/.
"""
from fastapi import APIRouter

from api.schemas import EDAInsight, EDASummaryResponse
from src.data.loader import eda_summary_from_db, load_application_train, table_path

router = APIRouter(prefix="/eda", tags=["eda"])

# One entry per chart in artifacts/eda/, in the same order and wording as the notebook's
# own summary — the API is a window onto that analysis, not a second copy of it.
INSIGHTS = [
    {
        "title": "Most missing data is in optional building/apartment fields",
        "description": "The top missing-value columns are almost entirely apartment/building details. Expected, since many applicants rent or live with family and simply have nothing to report there, not a data error.",
        "chart_url": "/charts/01_missing_values_top20.png",
    },
    {
        "title": "Only ~8% of applicants had trouble paying",
        "description": "A real class imbalance. A lazy model could predict 'no trouble' every time and still look 92% accurate while being useless. Drives the class-imbalance handling in the model.",
        "chart_url": "/charts/02_target_imbalance.png",
    },
    {
        "title": "External credit scores are the strongest single predictor",
        "description": "All three EXT_SOURCE scores are clearly lower, on average, for applicants who had trouble paying. Likely the single most informative signal in the dataset.",
        "chart_url": "/charts/03_ext_source_by_target.png",
    },
    {
        "title": "Younger applicants have more payment trouble than older ones",
        "description": "Default rate drops steadily with age. The 18-25 group runs at roughly double the rate of the 56-65 group, a simple, explainable risk factor.",
        "chart_url": "/charts/04_default_rate_by_age.png",
    },
    {
        "title": "Income type and education level both separate risk cleanly",
        "description": "Unemployed and maternity-leave applicants, along with lower-education segments, run well above the average trouble rate. Clean, policy-friendly segments.",
        "chart_url": "/charts/05_income_education_default_rate.png",
    },
    {
        "title": "Loan-to-income ratio matters, but only as one signal among several",
        "description": "Applicants who defaulted typically borrowed a bigger multiple of their income. Real, but a modest effect on its own compared to credit score or job type.",
        "chart_url": "/charts/06_credit_income_ratio_by_target.png",
    },
    {
        "title": "A data-quality fix turned into a real feature",
        "description": "The DAYS_EMPLOYED placeholder value (365243 means 'no current job') actually flags a lower-risk group, mostly retirees with stable pension income.",
        "chart_url": "/charts/07_days_employed_anomaly_default_rate.png",
    },
    {
        "title": "Having zero credit bureau history is itself a mild risk signal",
        "description": "Applicants with no bureau record on file default more than those with some track record. 'Unknown risk' isn't the same as 'low risk'.",
        "chart_url": "/charts/08_bureau_history_default_rate.png",
    },
]


@router.get("/summary", response_model=EDASummaryResponse)
def summary():
    if not table_path("application_train").exists():
        return EDASummaryResponse(**eda_summary_from_db())

    app = load_application_train()
    missing_counts = app.isna().sum()
    return EDASummaryResponse(
        n_applicants=len(app),
        n_features=app.shape[1],
        default_rate_pct=round(float(app["TARGET"].mean()) * 100, 2),
        n_columns_with_missing=int((missing_counts > 0).sum()),
    )


@router.get("/insights", response_model=list[EDAInsight])
def insights():
    return [EDAInsight(**item) for item in INSIGHTS]
