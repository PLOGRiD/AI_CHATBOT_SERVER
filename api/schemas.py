from typing import Literal

from pydantic import BaseModel, HttpUrl

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatResponse(BaseModel):
    answer: str | None

class ImageUrlRequest(BaseModel):
    image_url: HttpUrl

class ImageDisposalResponse(BaseModel):
    answer: str