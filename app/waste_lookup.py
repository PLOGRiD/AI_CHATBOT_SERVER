from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.vision import analyze_image_url
from prompt.waste_disposal_prompt import DISPOSAL_ANSWER_SYSTEM_PROMPT
from tool.waste_disposal import resolve_disposal_by_item

load_dotenv()

_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


async def _generate_answer(raw_text: str) -> str:
    response = await _MODEL.ainvoke(
        [SystemMessage(content=DISPOSAL_ANSWER_SYSTEM_PROMPT), HumanMessage(content=raw_text)]
    )
    return response.content


async def get_disposal_by_image_url(image_url: str) -> str:
    analysis = await analyze_image_url(image_url)

    if analysis.item_name is None:
        raw_text = "그건 분리배출 안내 대상이 아닌 것 같습니다."
    else:
        raw_text = await resolve_disposal_by_item(
            [analysis.item_name], analysis.category, analysis.subcategory
        )

    return await _generate_answer(raw_text)
