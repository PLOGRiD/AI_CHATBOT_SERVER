import json

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.constants import CATEGORIES, RECYCLABLE_SUBCATEGORIES
from prompt.waste_disposal_prompt import (
    ITEM_EXTRACT_SYSTEM_PROMPT,
    MATCH_VERIFY_SYSTEM_PROMPT,
)

load_dotenv()

_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
_EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

_CHROMA_PATH = "rag/chroma_db"
_COLLECTION_NAME = "waste_disposal_items"
_vectorstore = Chroma(
    collection_name=_COLLECTION_NAME,
    embedding_function=_EMBEDDINGS_MODEL,
    persist_directory=_CHROMA_PATH,
    collection_metadata={"hnsw:space": "cosine"},
)

_ITEMS_PATH = "rag/data/waste_items.json"
with open(_ITEMS_PATH, encoding="utf-8") as f:
    _ITEMS = json.load(f)


def _normalize(name):
    return name.replace(" ", "")


_ITEM_INDEX = {item["품목명"]: item for item in _ITEMS}
_ITEM_INDEX_NORMALIZED = {_normalize(item["품목명"]): item for item in _ITEMS}
_ALIAS_INDEX = {
    _normalize(synonym): item for item in _ITEMS for synonym in item.get("유의어", [])
}

_TYPE_GUIDE_PATH = "rag/data/waste_disposal_source.json"
with open(_TYPE_GUIDE_PATH, encoding="utf-8") as f:
    _TYPE_GUIDE = json.load(f)


class _ExtractedItem(BaseModel):
    item_names: list[str] = Field(
        default_factory=list,
        description="사용자가 배출 방법을 물어보는 품목명 또는 대상의 표현 후보 2~3개. 분리배출 대상이 될 수 없으면 빈 리스트",
    )
    category: str = Field(description=f"{CATEGORIES} 중 하나")
    subcategory: str | None = Field(
        default=None,
        description=f"category가 '재활용폐기물'인 경우 {RECYCLABLE_SUBCATEGORIES} 중 하나",
    )


class _MatchVerification(BaseModel):
    is_match: bool = Field(description="사용자가 물어본 품목과 검색된 품목이 실제로 같은 대상이면 true, 다르면 false")


def _format_item(item):
    return (
        f"품목명: {item['품목명']}\n"
        f"배출방법: {item['배출방법']}\n"
        f"특징: {item['특징']}"
    )


def _format_matched_doc(doc):
    metadata = doc.metadata
    return (
        f"품목명: {metadata['item_name']}\n"
        f"배출방법: {metadata['disposal_method']}\n"
        f"특징: {metadata['features']}"
    )


def _format_type_guide(category, subcategory=None):
    if category == "재활용폐기물":
        guide = _TYPE_GUIDE.get(category, {}).get("subcategories", {}).get(subcategory)
        if not guide:
            return None
        lines = [f"카테고리: {category} > {subcategory}"]
    else:
        guide = _TYPE_GUIDE.get(category)
        if not guide:
            return None
        lines = [f"카테고리: {category}"]

    if guide.get("배출방법_공통"):
        steps = "\n".join(f"- {s}" for s in guide["배출방법_공통"])
        lines.append(f"일반 배출 원칙:\n{steps}")

    for name, steps in guide.get("배출방법_세부", {}).items():
        step_lines = "\n".join(f"  {s}" for s in steps)
        lines.append(f"{name}:\n{step_lines}")

    return "\n\n".join(lines)


async def _verify_match(query, matched_name):
    structured_model = _MODEL.with_structured_output(_MatchVerification)
    result: _MatchVerification = await structured_model.ainvoke(
        [
            SystemMessage(content=MATCH_VERIFY_SYSTEM_PROMPT),
            HumanMessage(content=f"사용자가 물어본 품목: {query}\n검색된 품목: {matched_name}"),
        ]
    )
    return result.is_match


async def _resolve_context(item_names, category, subcategory):
    """
    1차: 품목명 정확 매칭
    - LLM이 뽑은 후보(item_names)를 순서대로 waste_items.json 인덱스와 대조한다.
    - 원문 그대로 먼저 대조 후, 안 맞으면 공백을 지운 버전으로 한 번 더 대조한다.(예: "우유 팩" -> "우유팩").
    - 정식 품목명으로 안 맞으면 유의어 사전과도 대조한다.
    - 카탈로그에 없는 품목(오탈자가 심하거나 아예 다른 표현)은 2차로 넘어간다.
    """
    for item_name in item_names:
        normalized = _normalize(item_name)
        item = (
            _ITEM_INDEX.get(item_name)
            or _ITEM_INDEX_NORMALIZED.get(normalized)
            or _ALIAS_INDEX.get(normalized)
        )
        if item:
            return _format_item(item)

    """
    2차: 임베딩 유사도 검색
    - 후보 전체를 각각 벡터DB에 검색해서 점수 높은 순으로 정렬한다.
    - 점수가 가장 높은 것부터 CRAG 검증(_verify_match, LLM에게 다시 한번 확인)을 시도한다.
    - CRAG 검증에서 true가 나오면 그 품목의 배출 방법을 안내한다.
    - CRAG 검증에서 false가 나오면 다음 후보로 넘어간다.
    """
    if item_names:
        candidates = []
        for candidate in item_names:
            matches = _vectorstore.similarity_search_with_relevance_scores(candidate, k=1)
            if matches:
                candidates.append((candidate, matches[0][0], matches[0][1]))

        candidates.sort(key=lambda c: c[2], reverse=True)
        for query, doc, _score in candidates:
            top_name = doc.metadata.get("item_name")
            if await _verify_match(query, top_name):
                return _format_matched_doc(doc)

    """
    3차: 카테고리 일반 가이드
    - 특정 품목 데이터는 못 찾았지만 category(재활용폐기물이면 subcategory까지)는 알고 있는 경우,
    - waste_disposal_source.json에서 그 카테고리의 일반적인 배출 원칙을 안내하여, 카테고리 수준의 안전한 답을 준다.
    """
    if category:
        guide = _format_type_guide(category, subcategory)
        if guide:
            return guide

    return None


@tool
async def waste_disposal_lookup(item_description: str) -> str:
    """사용자가 버리려는 물건의 분리배출/재활용 방법을 조회한다.

    물건 이름(예: "우유팩")이나, 정확한 이름을 모르면 형태·재질·색상 묘사(예: "투명 플라스틱 조각")를
    인자로 받는다. 특정 물건의 분리배출 방법을 묻는 질문에 사용하라.
    """
    structured_model = _MODEL.with_structured_output(_ExtractedItem)
    extracted: _ExtractedItem = await structured_model.ainvoke(
        [SystemMessage(content=ITEM_EXTRACT_SYSTEM_PROMPT), HumanMessage(content=item_description)]
    )

    if extracted.category == "해당없음":
        return "그건 분리배출 안내 대상이 아닌 것 같습니다."

    context_text = await _resolve_context(extracted.item_names, extracted.category, extracted.subcategory)
    return context_text or "해당 품목의 분리배출 정보를 찾지 못했습니다."
