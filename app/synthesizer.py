import json
from typing import AsyncGenerator
from .models import StateTracker, Message
from .providers.base import BaseLLMProvider

def _build_synthesis_prompt(state: StateTracker) -> str:
    """Combines the execution history into a single context string."""
    history = ""
    for task in state.execution_history:
        history += f"--- Task {task.task_id} ---\n"
        history += f"Description: {task.description}\n"
        history += f"Result: {json.dumps(task.result)}\n\n"
    prompt = (
        "You are the final synthesizer agent. The system has completed a series of tasks "
        "to answer the user's original request. Use the following execution results to "
        "provide a clear, comprehensive final answer to the user.\n\n"
        "EXECUTION RESULTS:\n"
        f"{history if history else 'No tasks were executed. Answer based on general knowledge.'}\n\n"
        "Synthesize the data and answer the user directly and naturally."
    )
    return prompt

async def stream_final_response(state: StateTracker, llm_provider: BaseLLMProvider) -> AsyncGenerator[str, None]:
    """Generates Server-Sent Events (SSE) from the LLM stream."""
    try:
        system_prompt = _build_synthesis_prompt(state)
        messages = [Message(role="system", content=system_prompt)]
        messages.extend(state.chat_request.messages)
        async for chunk in llm_provider.stream_text(messages):
            if isinstance(chunk, dict) and "error" in chunk:
                yield f"data: {json.dumps(chunk)}\n\n"
                break
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    except Exception as e:
        error_dict = {
            "message": "A critical error occurred during the final synthesis streaming.",
            "error": str(e),
            "status_code": 500
        }
        yield f"data: {json.dumps(error_dict)}\n\n"