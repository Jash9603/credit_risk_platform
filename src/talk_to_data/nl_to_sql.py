"""Orchestrates one talk-to-data turn: rewrite -> generate SQL -> validate -> execute ->
synthesise a plain-English answer, with a bounded retry loop when the generated SQL is
invalid.

Conversation history is owned entirely by query_rewriter.rewrite_question — every step
after that works with a single standalone question and never touches history directly.
See query_rewriter.py for why (it's also the token-optimisation story for multi-turn
conversations).

Run: `python -m src.talk_to_data.nl_to_sql` for an interactive CLI.
"""
import json
from pathlib import Path
from typing import Optional

from src.talk_to_data.llm_client import chat_completion
from src.talk_to_data.prompt_templates import ANSWER_SYSTEM_PROMPT, PROMPT_VERSION, get_sql_system_prompt
from src.talk_to_data.query_rewriter import rewrite_question
from src.talk_to_data.query_runner import QueryExecutionError, run_query
from src.talk_to_data.sql_validator import SQLValidationError, validate_sql
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2


def _has_live_key() -> bool:
    return bool(settings.openai_api_key)


def ask(question: str, history: list = None) -> dict:
    """Returns {question, standalone_question, sql, rows, answer, refused}. `history` is
    a list of {"role": "user"|"assistant", "content": ...} from earlier turns in the same
    conversation — pass the growing list back in on each call for follow-up questions.
    History is only ever read here, by rewrite_question; nothing downstream sees it."""
    history = history or []
    logger.info("nl_to_sql turn (prompt_version=%s): %s", PROMPT_VERSION, question)

    if not _has_live_key():
        return {
            "question": question, "standalone_question": question, "sql": None, "rows": [],
            "answer": "No LLM key is configured. Set an API key in .env to ask anything freely.",
            "refused": True,
        }

    standalone_question = rewrite_question(question, history)

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        prompt_input = standalone_question if attempt == 0 else (
            f"{standalone_question}\n\nYour previous SQL was invalid: {last_error}\nFix it and try again."
        )
        raw = chat_completion(get_sql_system_prompt(), prompt_input, []).strip()
        raw = raw.removeprefix("SQL:").strip().strip("`").strip()

        if raw.startswith("NO_SQL:"):
            return {"question": question, "standalone_question": standalone_question, "sql": None,
                     "rows": [], "answer": raw[len("NO_SQL:"):].strip(), "refused": True}

        try:
            safe_sql = validate_sql(raw)
        except SQLValidationError as e:
            last_error = str(e)
            logger.warning("SQL validation failed (attempt %d): %s", attempt + 1, e)
            continue

        try:
            result = run_query(safe_sql)
        except QueryExecutionError as e:
            last_error = str(e)
            logger.warning("SQL execution failed (attempt %d): %s", attempt + 1, e)
            continue

        answer = synthesize_answer(standalone_question, result)
        return {"question": question, "standalone_question": standalone_question, "sql": safe_sql,
                "rows": result["rows"], "answer": answer, "refused": False}

    return {
        "question": question, "standalone_question": standalone_question, "sql": None, "rows": [],
        "answer": f"I couldn't produce a valid query after {MAX_RETRIES + 1} attempts (last error: {last_error}).",
        "refused": True,
    }


def synthesize_answer(question: str, result: dict) -> str:
    if not result["rows"]:
        return "The query ran successfully but returned no rows."
    context = f"Question: {question}\nRows: {json.dumps(result['rows'][:50], default=str)}"
    return chat_completion(ANSWER_SYSTEM_PROMPT, context, [])


if __name__ == "__main__":
    print("Talk to your credit risk data. Type 'exit' to quit.\n")
    conversation: list = []
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        result = ask(question, conversation)
        print(f"\nSQL: {result['sql']}")
        print(f"Answer: {result['answer']}\n")
        if not result["refused"]:
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": result["answer"]})
