"""OpenAI client wrapper for the talk-to-data chatbot."""
from openai import OpenAI

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def chat_completion(system_prompt: str, user_message: str, history: list = None) -> str:
    history = history or []
    client = OpenAI(api_key=settings.openai_api_key)

    messages = [{"role": "system", "content": system_prompt}, *history,
                {"role": "user", "content": user_message}]
    response = client.chat.completions.create(
        model=settings.llm_model, messages=messages, temperature=0,
    )
    return response.choices[0].message.content.strip()
