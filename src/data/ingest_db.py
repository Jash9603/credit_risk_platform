"""Idempotent CSV -> PostgreSQL loader for the talk-to-data database.

Streams each CSV straight into Postgres via COPY through the client connection (works
identically whether Postgres is local or containerized) rather than row-by-row INSERTs,
which would be far too slow for the ~20M-row behaviour tables. The COPY column list is
built from the CSV's own header, lower-cased, so it doesn't matter that schema.sql defines
lowercase columns while the source files are UPPER_SNAKE_CASE — no data rewriting needed.

Run: `python -m src.data.ingest_db`. Safe to re-run: any table that already has rows is
skipped, so a crashed or repeated run never double-loads data.
"""
import csv

import psycopg2

from src.data.loader import TABLE_FILES, table_path
from src.utils.config import settings
from src.utils.docker_utils import wait_for_port
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _connect():
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


def _table_is_empty(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(f"SELECT 1 FROM {table} LIMIT 1;")
        return cur.fetchone() is None


def _csv_header(path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def ingest_table(conn, table: str) -> None:
    if not _table_is_empty(conn, table):
        logger.info("Skipping %s — already loaded", table)
        return

    path = table_path(table)
    columns = [c.lower() for c in _csv_header(path)]
    copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, HEADER true)"

    with conn.cursor() as cur, open(path, "r", encoding="utf-8", newline="") as f:
        cur.copy_expert(copy_sql, f)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
    logger.info("Loaded %s: %s rows", table, count)


def main() -> None:
    wait_for_port(settings.postgres_host, settings.postgres_port, timeout=60)
    conn = _connect()
    try:
        for table in TABLE_FILES:
            ingest_table(conn, table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
