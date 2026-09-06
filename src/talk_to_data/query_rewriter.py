"""The node that owns conversation history for the talk-to-data chatbot.

Every other step (SQL generation, validation, execution, answer synthesis) works with a
single standalone question and never sees the conversation history directly. This node is
the one place history is read: it takes the growing conversation plus a new, possibly
context-dependent follow-up ("show me the same thing but by income type instead") and
produces one fully self-contained question ("show the default rate by income type").

This is also the token-optimisation story for multi-turn conversations: without this
step, every SQL-generation call would need the *entire* growing history appended
alongside the schema-heavy SQL prompt, so token cost climbs with every turn taken. With
it, the history is read once, by a small prompt that carries no schema at all, and every
downstream step only ever pays for one short, standalone question.
"""
from src.talk_to_data.llm_client import chat_completion
from src.talk_to_data.prompt_templates import REWRITE_SYSTEM_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rewrite_question(question: str, history: list) -> str:
    if not history:
        return question

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    prompt_input = f"Conversation so far:\n{transcript}\n\nNew follow-up question: {question}"

    rewritten = chat_completion(REWRITE_SYSTEM_PROMPT, prompt_input, []).strip()
    if rewritten and rewritten != question:
        logger.info("Rewrote follow-up question: %r -> %r", question, rewritten)
    return rewritten or question
