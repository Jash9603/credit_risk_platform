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
import io
import itertools

import psycopg2

from src.data.loader import TABLE_FILES, table_path
from src.utils.config import settings
from src.utils.docker_utils import wait_for_port
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rows per COPY + commit. A local/plain Postgres container is reliable and happy to take
# one giant transaction (a large chunk just gives crash-resumability without much cost).
# A managed/serverless backend (Neon, CockroachDB) is a different story: it tracks
# per-row locks for the life of a transaction with a fixed budget that a several-hundred-
# thousand-row single transaction blows past, and its proxy can drop a long-lived
# connection mid-COPY for reasons that have nothing to do with the data. Small chunks
# there trade some throughput for actually finishing reliably; there's no such trade-off
# to make locally, so keep local ingestion fast instead of needlessly throttling it.
CHUNK_ROWS = 10_000 if settings.database_url else 2_000_000
MAX_ATTEMPTS = 5


def _connect():
    return psycopg2.connect(settings.pg_dsn)


def _table_is_empty(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(f"SELECT 1 FROM {table} LIMIT 1;")
        return cur.fetchone() is None


def _csv_header(path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _truncate(table: str) -> None:
    """Clean slate on a fresh connection — used both when a previous attempt left a
    partial load behind and before retrying, so the next attempt (or the next run's
    _table_is_empty check) never sees leftover rows from a dropped connection."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table};")
        conn.commit()
    finally:
        conn.close()


def ingest_table(table: str) -> None:
    conn = _connect()
    try:
        if not _table_is_empty(conn, table):
            logger.info("Skipping %s — already loaded", table)
            return
    finally:
        conn.close()

    path = table_path(table)
    columns = [c.lower() for c in _csv_header(path)]
    copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)"

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        conn = _connect()
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                next(f)  # header — already captured via _csv_header, not part of any chunk
                while True:
                    chunk = list(itertools.islice(f, CHUNK_ROWS))
                    if not chunk:
                        break
                    with conn.cursor() as cur:
                        cur.copy_expert(copy_sql, io.StringIO("".join(chunk)))
                    conn.commit()

            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
            logger.info("Loaded %s: %s rows", table, count)
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            last_error = exc
            logger.warning(
                "Connection dropped loading %s (attempt %s/%s): %s — retrying from scratch",
                table, attempt, MAX_ATTEMPTS, exc,
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # A managed/serverless backend can close a long-lived connection mid-COPY for
        # reasons that have nothing to do with the data (proxy recycling, etc). Rather
        # than try to resume mid-file (and risk skipping or duplicating rows), wipe
        # whatever chunks made it in and restart the table from row zero.
        _truncate(table)

    raise RuntimeError(f"Failed to load {table} after {MAX_ATTEMPTS} attempts") from last_error


def main() -> None:
    if not table_path("application_train").exists():
        logger.info(
            "No local CSVs found (production image) — assuming the target database "
            "was already seeded and skipping ingest."
        )
        return

    if not settings.database_url:
        wait_for_port(settings.postgres_host, settings.postgres_port, timeout=60)

    for table in TABLE_FILES:
        ingest_table(table)


if __name__ == "__main__":
    main()
