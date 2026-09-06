"""AST-level SQL safety checks using sqlglot.

This is defense-in-depth: even if a prompt is crafted to trick the LLM into generating
something destructive (or the model just makes a mistake), this validator runs on the
actual generated SQL text, independent of the prompt or the model. It doesn't trust the
LLM's own promise to "only write SELECT" — it verifies the parsed structure.
"""
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {
    "application_train", "bureau", "bureau_balance", "previous_application",
    "pos_cash_balance", "credit_card_balance", "installments_payments",
}
MAX_ROWS = 200


class SQLValidationError(Exception):
    """Raised with a message specific enough to feed back to the LLM for a retry."""


def validate_sql(sql: str) -> str:
    """Returns a safe, executable SQL string (with a row limit enforced), or raises
    SQLValidationError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except Exception as e:
        raise SQLValidationError(f"SQL did not parse: {e}")

    if len(statements) != 1:
        raise SQLValidationError("Exactly one SQL statement is allowed.")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise SQLValidationError("Only SELECT statements are allowed — no INSERT/UPDATE/DELETE/DDL.")

    tables = {t.name.lower() for t in stmt.find_all(exp.Table)}
    unknown = tables - ALLOWED_TABLES
    if unknown:
        raise SQLValidationError(
            f"Unknown or disallowed table(s): {', '.join(sorted(unknown))}. "
            f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}."
        )

    existing_limit = stmt.args.get("limit")
    if existing_limit is None:
        stmt = stmt.limit(MAX_ROWS)
    else:
        try:
            requested = int(existing_limit.expression.this)
            if requested > MAX_ROWS:
                stmt.set("limit", exp.Limit(expression=exp.Literal.number(MAX_ROWS)))
        except (AttributeError, ValueError):
            pass

    return stmt.sql(dialect="postgres")
