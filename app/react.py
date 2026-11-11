import json, re
from typing import AsyncGenerator, List, Dict, Any
from .models import StateTracker, Message
from .providers.base import BaseLLMProvider
from .tools.registry import tool_registry

# Claude's heuristic refusal regex
REFUSAL_REGEX = re.compile(
    r"\b(i\s+(cannot|can't|am\s+unable|am\s+not\s+able|don't\s+have|do\s+not\s+have)|unable\s+to|not\s+(directly\s+)?(possible|supported)|no\s+(tool|way|function)s?\s+(to|for|that))\b",
    re.IGNORECASE
)
REFUSAL_NUDGE = "You have tools available and have not used any of them. Do not refuse yet. Use the tools to fetch the data this needs, and then answer. If it truly cannot be done after trying, say what you tried and what is missing."

async def run_react_agent(state: StateTracker, llm_provider: BaseLLMProvider, available_tools: List[Dict[str, Any]], max_steps: int = 5) -> AsyncGenerator[str, None]:
    messages = state.chat_request.messages.copy()
    nudged = False
    
    for step in range(max_steps):
        agent_result = await llm_provider.invoke_tool_agent(messages, available_tools)
        if isinstance(agent_result, dict) and "error" in agent_result:
            yield f"data: {json.dumps(agent_result)}\n\n"
            return
        print("React Result\n", agent_result)
        state.track_usage(agent_result)
        if not agent_result.tool_calls and not nudged and available_tools:
            if agent_result.text and REFUSAL_REGEX.search(agent_result.text.strip().replace(" ", "'")):
                nudged = True
                messages.append(Message(role="assistant", content=agent_result.text))
                messages.append(Message(role="user", content=REFUSAL_NUDGE))
                continue
        if agent_result.tool_calls:
            if agent_result.text:
                yield f"data: {json.dumps({'type': 'text', 'chunk': agent_result.text + '\n\n'})}\n\n"
            for tc in agent_result.tool_calls:
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc.name, 'args': tc.args})}\n\n"
            messages.append(Message(role="assistant", tool_calls=agent_result.tool_calls))
            for tc in agent_result.tool_calls:
                execution_result = await tool_registry.execute(tc.name, tc.args)
                yield f"data: {json.dumps({'type': 'tool_result', 'name': tc.name, 'result': execution_result})}\n\n"
                messages.append(Message(
                    role="tool", 
                    tool_call_id=tc.id, 
                    tool_name=tc.name, 
                    tool_result=execution_result
                ))
            continue
        async for chunk in llm_provider.stream_text(messages):
            if isinstance(chunk, dict) and "error" in chunk:
                yield f"data: {json.dumps(chunk)}\n\n"
                break
            yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
        break
    yield f"data: {json.dumps({'type': 'done', 'input_tokens': state.total_input_tokens, 'output_tokens': state.total_output_tokens})}\n\n"