from fastapi import APIRouter

from api.schemas import ChatRequest, ChatResponse
from src.talk_to_data.nl_to_sql import ask

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = ask(request.question, history)
    return ChatResponse(**result)
