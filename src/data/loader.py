"""Load Home Credit CSVs as pandas DataFrames.

Used directly by the ML pipeline (which needs a full feature matrix and has no reason to
round-trip through SQL) and reused by build_schema.py / ingest_db.py, which populate an
independent Postgres copy for the talk-to-data chatbot to query. Two data paths, one
source of truth — see DECISIONS.md, "Data Loading & Ingest".
"""
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.utils.config import settings
from src.utils.helpers import reduce_mem_usage, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Table name -> source CSV. application_test / sample_submission are intentionally
# excluded: this platform scores applicants directly, it does not produce a Kaggle
# leaderboard file, so there is nothing for them to feed.
TABLE_FILES: dict[str, str] = {
    "application_train": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
}

# Stable identifiers, not measurements — always BIGINT downstream, never downcast to a
# lossy numeric type and never treated as a model feature.
ID_COLUMNS = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"}


def table_path(table: str) -> Path:
    if table not in TABLE_FILES:
        raise KeyError(f"Unknown table '{table}'. Known tables: {sorted(TABLE_FILES)}")
    return settings.data_path / TABLE_FILES[table]


def load_table(table: str, nrows: Optional[int] = None, reduce_memory: bool = True) -> pd.DataFrame:
    """Load one CSV fully into memory. Fine for every table except the three
    tens-of-millions-of-rows behaviour tables — use iter_table_chunks for those."""
    path = table_path(table)
    with timer(f"load_table({table})"):
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
    if reduce_memory:
        df = reduce_mem_usage(df, verbose=False)
    logger.info("Loaded %s: %s rows, %s columns", table, len(df), df.shape[1])
    return df


def iter_table_chunks(table: str, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Stream a large table in chunks instead of loading it whole."""
    path = table_path(table)
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        yield chunk


@lru_cache(maxsize=1)
def load_application_train() -> pd.DataFrame:
    """Convenience entry point for EDA/ML — the table every later phase starts from.

    Cached for the life of the process: safe because every caller in this codebase
    copies before mutating (see preprocessor.py), so nothing writes back into the shared
    cached frame. This is what makes a long-running process (the API server) load it once
    at startup instead of once per request — a single CLI invocation still only loads it
    once anyway, so this changes nothing for that path.

    The 2.5GB raw CSVs are never shipped in the deployed API image — falls back to
    reading the same table from Postgres (already seeded there for the chatbot) when the
    local CSV isn't present, so a CLI run or notebook still works unchanged in that
    environment. The live API deliberately avoids calling this function at all when
    there's no local CSV (see api/main.py, get_applicant_row, eda summary()) — a free-tier
    host doesn't have the RAM to hold all 307k rows resident, so production queries
    Postgres directly, per request, for the one row or the handful of aggregates it
    actually needs instead of materialising the whole table.

    Uses a raw psycopg2 connection rather than a SQLAlchemy engine: SQLAlchemy's Postgres
    dialect parses the server's version string on first connect, and CockroachDB's version
    string ("CockroachDB CCL v26.2.5 ...") doesn't match the "PostgreSQL X.Y" pattern it
    expects, raising a hard AssertionError. Raw psycopg2 skips that dialect layer entirely.
    """
    if table_path("application_train").exists():
        return load_table("application_train")

    import psycopg2

    logger.info("application_train.csv not found locally — loading it from Postgres instead")
    with psycopg2.connect(settings.pg_dsn) as conn:
        df = pd.read_sql("SELECT * FROM application_train", conn)
    df.columns = [c.upper() for c in df.columns]
    return reduce_mem_usage(df, verbose=False)


NUMERIC_PG_TYPES = {"double precision", "bigint", "smallint", "integer", "real", "numeric"}


def query_applicant_row(sk_id_curr: int) -> pd.DataFrame:
    """Single-applicant lookup straight from Postgres — the production equivalent of
    filtering load_application_train() by SK_ID_CURR, without ever pulling the other
    307,510 rows into memory to answer a one-row question.

    A single-row DataFrame can't infer a numeric dtype from one NULL value alone —
    pandas falls back to `object`, which prepare_categoricals() (preprocessor.py) then
    misreads as a categorical column, giving LightGBM one categorical column more than
    it was trained with ("train and valid dataset categorical_feature do not match").
    The CSV path never hits this: pandas.read_csv establishes each column's dtype from
    all 307k rows before any single row is filtered out. Explicitly casting every
    genuinely-numeric Postgres column here reproduces that same dtype, whether or not
    this one applicant's value happens to be NULL.
    """
    import psycopg2

    with psycopg2.connect(settings.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'application_train' ORDER BY ordinal_position"
            )
            schema = cur.fetchall()

            cur.execute("SELECT * FROM application_train WHERE sk_id_curr = %s", (sk_id_curr,))
            row = cur.fetchone()

    colnames = [name.upper() for name, _ in schema]
    if row is None:
        return pd.DataFrame(columns=colnames)

    df = pd.DataFrame([row], columns=colnames)
    for name, data_type in schema:
        if data_type in NUMERIC_PG_TYPES:
            df[name.upper()] = pd.to_numeric(df[name.upper()], errors="coerce")
    return df


def eda_summary_from_db() -> dict:
    """The four EDA summary numbers (applicant count, feature count, default rate,
    columns with any missing value), computed with one aggregate SQL query instead of
    pulling all 307k rows client-side just to call .isna().sum() on them."""
    import psycopg2

    with psycopg2.connect(settings.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'application_train'"
            )
            columns = [r[0] for r in cur.fetchall()]

            null_exprs = ", ".join(f'SUM(("{c}" IS NULL)::INT) AS "{c}"' for c in columns)
            cur.execute(f"SELECT COUNT(*) AS n, AVG(target) AS rate, {null_exprs} FROM application_train")
            row = cur.fetchone()
            colnames = [d[0] for d in cur.description]

    values = dict(zip(colnames, row))
    n_applicants = values.pop("n")
    default_rate = float(values.pop("rate"))
    n_columns_with_missing = sum(1 for v in values.values() if v)

    return {
        "n_applicants": n_applicants,
        "n_features": len(columns),
        "default_rate_pct": round(default_rate * 100, 2),
        "n_columns_with_missing": n_columns_with_missing,
    }
