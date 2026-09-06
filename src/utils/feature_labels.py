"""Plain-English feature names, shared by SHAP explanations and rule derivation so both
describe the same feature to a non-technical reader the same way."""

FEATURE_LABELS = {
    "EXT_SOURCE_1": "external credit score #1",
    "EXT_SOURCE_2": "external credit score #2",
    "EXT_SOURCE_3": "external credit score #3",
    "AGE_YEARS": "applicant's age",
    "YEARS_EMPLOYED": "years at current job",
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "loan amount",
    "CREDIT_INCOME_RATIO": "loan amount relative to income",
    "ANNUITY_INCOME_RATIO": "monthly payment relative to income",
    "GOODS_CREDIT_RATIO": "loan amount relative to goods price",
    "N_BUREAU_LOANS": "number of other loans on record",
    "N_BUREAU_BAD_DEBT": "number of bad-debt records with the credit bureau",
    "N_PREV_REFUSED": "number of past refused applications",
    "N_PREV_APPLICATIONS": "number of past applications with Home Credit",
    "IS_DAYS_EMPLOYED_ANOMALY": "no current job on record",
    "CODE_GENDER": "gender",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income type",
    "NAME_FAMILY_STATUS": "family status",
}


def readable_name(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").lower())


def format_value(value) -> str:
    # np.float32 (from reduce_mem_usage downcasting) isn't a Python `float` subclass,
    # so check for "can this be treated as a float" rather than the exact built-in type.
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
