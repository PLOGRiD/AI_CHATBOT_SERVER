from langchain_core.messages import HumanMessage

from app.agent import run_agent
from app.vision import analyze_image


async def run_chat(
    session_id: str,
    message: str | None,
    image_bytes: bytes | None,
) -> str:
    context_parts = []

    if image_bytes is not None:
        analysis = await analyze_image(image_bytes)
        subcategory_part = f" > {analysis.subcategory}" if analysis.subcategory else ""
        context_parts.append(
            "[첨부 이미지 분석 결과]\n"
            f"품목명: {analysis.item_name}\n"
            f"카테고리: {analysis.category}{subcategory_part}\n"
            f"이미지 설명: {analysis.description}"
        )

    if message:
        context_parts.append(message)

    user_content = "\n\n".join(context_parts) if context_parts else "(첨부된 이미지를 분석해줘)"

    return await run_agent([HumanMessage(content=user_content)])
