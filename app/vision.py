import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.constants import CATEGORIES, RECYCLABLE_SUBCATEGORIES
from prompt.vision_prompt import VISION_SYSTEM_PROMPT

load_dotenv()

_VISION_MODEL = ChatOpenAI(model="gpt-5", temperature=0)


class ImageAnalysis(BaseModel):
    item_name: str | None = Field(
        default=None,
        description="이미지 속 쓰레기의 품목명을 작성, 정확한 품목을 모르면 형태·재질·색상 기반으로 구체적으로 묘사, 실제로 폐기할 수 있는 항목이 아니면 null",
    )
    category: str = Field(description=f"{CATEGORIES} 중 하나")
    subcategory: str | None = Field(
        default=None,
        description=f"category가 '재활용폐기물'인 경우 {RECYCLABLE_SUBCATEGORIES} 중 하나",
    )
    description: str = Field(
        description="이미지에 보이는 것에 대한 시각적 묘사만 (분류 이유, 배출 방법 등 부가 설명 금지)"
    )


async def analyze_image(image_bytes: bytes) -> ImageAnalysis:
    structured_model = _VISION_MODEL.with_structured_output(ImageAnalysis)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    result: ImageAnalysis = await structured_model.ainvoke(
        [
            SystemMessage(content=VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                ]
            ),
        ]
    )
    return result


async def analyze_image_url(image_url: str) -> ImageAnalysis:
    structured_model = _VISION_MODEL.with_structured_output(ImageAnalysis)

    result: ImageAnalysis = await structured_model.ainvoke(
        [
            SystemMessage(content=VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                ]
            ),
        ]
    )
    return result
