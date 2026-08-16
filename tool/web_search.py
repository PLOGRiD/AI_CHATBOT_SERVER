from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

_search = TavilySearch(max_results=5, search_depth="advanced")


@tool
async def web_search(query: str) -> str:
    """분리배출·환경 정책 관련 최신 정보나 뉴스가 필요할 때 웹을 검색한다.

    특정 품목의 분리배출 방법에 대한 질의는 waste_disposal_lookup만으로 해결해야하며, 이 툴은 그걸로 해결되지 않는
    최신 정보·정책·뉴스 검색이 필요할 때만 사용하라.
    """
    search_results = await _search.ainvoke({"query": query})

    content = "\n\n".join(
        f"제목: {r['title']}\n출처: {r['url']}\n내용: {r['content']}"
        for r in search_results.get("results", [])
    )
    return content or "검색 결과가 없습니다."