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
    local CSV isn't present, so the identical code path works locally and in production.
    """
    if table_path("application_train").exists():
        return load_table("application_train")

    from sqlalchemy import create_engine

    logger.info("application_train.csv not found locally — loading it from Postgres instead")
    engine = create_engine(settings.pg_dsn)
    df = pd.read_sql("SELECT * FROM application_train", engine)
    df.columns = [c.upper() for c in df.columns]
    return reduce_mem_usage(df, verbose=False)
