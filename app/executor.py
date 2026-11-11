import json
from typing import List, Dict, Any, AsyncGenerator
from .models import StateTracker, Message
from .providers.base import BaseLLMProvider
from .tools.registry import tool_registry

def _build_execution_context(state: StateTracker) -> str:
    """Builds a summary of previously completed tasks to feed into the current step."""
    if not state.execution_history:
        return "No tasks have been completed yet."
        
    context = "COMPLETED TASKS HISTORY:\n"
    for completed_task in state.execution_history:
        context += f"- Task {completed_task.task_id} ({completed_task.description}):\n"
        context += f"  Result: {json.dumps(completed_task.result)}\n"
    return context

async def execute_plan(state: StateTracker, llm_provider: BaseLLMProvider, available_tools: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    try:
        if not state.plan or not state.plan.tasks:
            yield f"data: {json.dumps({'type': 'info', 'message': 'No tasks to execute.'})}\n\n"
            return
        for task in state.plan.tasks:
            task.status = "in_progress"
            yield f"data: {json.dumps({'type': 'task_start', 'task_id': task.task_id, 'description': task.description})}\n\n"
            system_prompt = (
                f"You are executing Task {task.task_id}: {task.description}\n\n"
                f"{_build_execution_context(state)}\n\n"
                "Evaluate the task and use the available tools to complete it."
            )
            messages = [Message(role="system", content=system_prompt)]
            messages.extend(state.chat_request.messages)
            agent_result = await llm_provider.invoke_tool_agent(messages, available_tools)
            if isinstance(agent_result, dict) and "error" in agent_result:
                yield f"data: {json.dumps({'type': 'error', 'details': agent_result})}\n\n"
                return
            print(agent_result)
            state.track_usage(agent_result)

            if agent_result.text:
                yield f"data: {json.dumps({'type': 'text', 'chunk': agent_result.text + '\n\n'})}\n\n"

            if not agent_result.tool_calls:
                state.mark_task_complete(task.task_id, result={"text_output": agent_result.text})
                continue
            task_results = []
            for tc in agent_result.tool_calls:
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc.name, 'args': tc.args})}\n\n"
                execution_result = await tool_registry.execute(tc.name, tc.args)
                yield f"data: {json.dumps({'type': 'tool_result', 'name': tc.name, 'result': execution_result})}\n\n"

                task_results.append({
                    "tool": tc.name,
                    "args": tc.args,
                    "output": execution_result
                })
            state.mark_task_complete(task.task_id, result=task_results)

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Execution loop failed', 'details': str(e)})}\n\n"