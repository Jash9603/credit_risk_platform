"""Execute a validated SQL query against Postgres and return plain Python data.

Two safety nets beyond the AST-level validator: a statement timeout (a query that somehow
slips through as expensive/slow gets killed rather than tying up the connection) and an
explicit `READ ONLY` transaction, which Postgres itself enforces at the SQL level — so
even a bug in the validator can't turn into a write against the real data.
"""
import psycopg2
import psycopg2.extras

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATEMENT_TIMEOUT_MS = 5000


class QueryExecutionError(Exception):
    pass


def run_query(sql: str) -> dict:
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
        return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}
    except psycopg2.Error as e:
        raise QueryExecutionError(str(e).strip())
    finally:
        conn.close()
