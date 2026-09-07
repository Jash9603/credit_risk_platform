"""Full database schema description for the LLM prompt.

Column listings are generated from the real Postgres schema (information_schema), not
hand-typed — same principle as sql/schema.sql itself: read the real thing rather than
transcribe it and risk it going stale. Every real column across all 7 tables is included,
so joins and questions spanning the whole dataset work, not just a hand-picked subset.

A column list alone can't convey everything a correct query needs, so a small
supplementary block adds the two things that actually cause wrong SQL in practice:
the exact string values a categorical column holds (guessing "University" when the real
value is "Higher education" silently returns zero rows, no error), and the join keys
linking the tables.
"""
import psycopg2

from src.utils.config import settings

TABLES = [
    "application_train", "bureau", "bureau_balance", "previous_application",
    "pos_cash_balance", "credit_card_balance", "installments_payments",
]

SUPPLEMENTARY_NOTES = """
Known categorical values (use these exact strings in WHERE clauses — if the user's
wording is close but not an exact match, e.g. a misspelling or a synonym, use the closest
value from this list rather than the user's own wording, since anything else silently
matches zero rows instead of raising an error):
  application_train.code_gender: 'M', 'F'
  application_train.name_education_type: 'Academic degree', 'Higher education',
    'Incomplete higher', 'Lower secondary', 'Secondary / secondary special'
  application_train.name_income_type: 'Working', 'Commercial associate', 'Pensioner',
    'State servant', 'Unemployed', 'Student', 'Businessman', 'Maternity leave'
  application_train.name_contract_type: 'Cash loans', 'Revolving loans'
  application_train.flag_own_car / flag_own_realty: 'Y', 'N'
  bureau.credit_active: 'Active', 'Closed', 'Sold', 'Bad debt'
  previous_application.name_contract_status: 'Approved', 'Refused', 'Canceled', 'Unused offer'
  previous_application.name_contract_type: 'Cash loans', 'Consumer loans',
    'Revolving loans', 'XNA'

Known data quirks:
  application_train.days_employed: negative days; the sentinel value 365243 means
    "no current job" (retired/unemployed), not a real value — filter it out explicitly
    if computing an average or similar over this column.
  Every days_* column is a negative day-offset from the application date
    (e.g. age in years = -days_birth / 365.25).

Join keys:
  application_train.sk_id_curr = bureau.sk_id_curr            (one applicant, many rows)
  application_train.sk_id_curr = previous_application.sk_id_curr (one applicant, many rows)
  bureau.sk_id_bureau = bureau_balance.sk_id_bureau
  previous_application.sk_id_prev = pos_cash_balance.sk_id_prev
                                   = credit_card_balance.sk_id_prev
                                   = installments_payments.sk_id_prev
"""


def _fetch_columns(table: str) -> list:
    conn = psycopg2.connect(settings.pg_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = %s ORDER BY ordinal_position",
                (table,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def build_schema_card() -> str:
    lines = ["Tables (PostgreSQL, lowercase snake_case names) — every real column is listed:\n"]
    for table in TABLES:
        columns = _fetch_columns(table)
        col_text = ", ".join(f"{name} ({dtype})" for name, dtype in columns)
        lines.append(f"{table} ({len(columns)} columns):\n  {col_text}\n")
    lines.append(SUPPLEMENTARY_NOTES)
    return "\n".join(lines)


_cached_card = None


def get_schema_card() -> str:
    """Fetched from Postgres once per process and cached — not at import time, so
    nothing here creates a database dependency for a code path that doesn't need it."""
    global _cached_card
    if _cached_card is None:
        _cached_card = build_schema_card()
    return _cached_card
