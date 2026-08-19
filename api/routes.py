import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import TypeAdapter, ValidationError

from api.schemas import ChatMessage, ChatResponse
from app.chat import run_chat

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

_HISTORY_ADAPTER = TypeAdapter(list[ChatMessage])

@router.post("", response_model=ChatResponse)
async def chat(
    message: str | None = Form(None),
    image: UploadFile | None = File(None),
    history: str | None = Form(None),
) -> ChatResponse:
    if not message and not image:
        raise HTTPException(status_code=400, detail="message 또는 image 중 하나는 필수입니다.")

    chat_history: list[ChatMessage] = []
    if history:
        try:
            chat_history = _HISTORY_ADAPTER.validate_json(history)
        except (json.JSONDecodeError, ValidationError):
            raise HTTPException(status_code=400, detail="history 필드는 [{role, content}] 형식의 JSON이어야 합니다.")

    image_bytes: bytes | None = None
    if image is not None:
        if not (image.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="image 필드는 이미지 파일이어야 합니다.")
        image_bytes = await image.read()

    answer = await run_chat(
        message=message,
        image_bytes=image_bytes,
        history=[turn.model_dump() for turn in chat_history],
    )

    return ChatResponse(answer=answer)
