from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from prompt.agent_prompt import AGENT_SYSTEM_PROMPT
from tool.waste_disposal import waste_disposal_lookup
from tool.web_search import web_search

load_dotenv()

_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
_TOOLS = [waste_disposal_lookup, web_search]
_TOOLS_BY_NAME = {t.name: t for t in _TOOLS}
_MODEL_WITH_TOOLS = _MODEL.bind_tools(_TOOLS)

_MAX_STEPS = 4

async def run_agent(messages: list) -> str:
    conversation = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]

    for _ in range(_MAX_STEPS):
        response: AIMessage = await _MODEL_WITH_TOOLS.ainvoke(conversation)
        conversation.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_fn = _TOOLS_BY_NAME[tool_call["name"]]
            result = await tool_fn.ainvoke(tool_call["args"])
            conversation.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    final: AIMessage = await _MODEL.ainvoke(conversation)
    return final.content
