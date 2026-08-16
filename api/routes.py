from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas import ChatResponse
from app.chat import run_chat

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(
    session_id: str = Form(...),
    message: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> ChatResponse:
    if not message and not image:
        raise HTTPException(status_code=400, detail="message 또는 image 중 하나는 필수입니다.")

    image_bytes: bytes | None = None
    if image is not None:
        if not (image.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="image 필드는 이미지 파일이어야 합니다.")
        image_bytes = await image.read()

    answer = await run_chat(
        session_id=session_id,
        message=message,
        image_bytes=image_bytes,
    )

    return ChatResponse(session_id=session_id, answer=answer)
