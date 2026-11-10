import json
from typing import List, Dict, Any
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

async def execute_plan(state: StateTracker, llm_provider: BaseLLMProvider, available_tools: List[Dict[str, Any]]) -> Any:
    """
    Iterates through the generated plan, evaluates each task, and executes tools.
    """
    try:
        if not state.plan or not state.plan.tasks:
            return {"status": "success", "message": "No tasks to execute."}

        for task in state.plan.tasks:
            task.status = "in_progress"
            system_prompt = (
                f"You are executing Task {task.task_id}: {task.description}\n\n"
                f"{_build_execution_context(state)}\n\n"
                "Evaluate the task and use the available tools to complete it. "
                "If no tool is needed, provide the reasoned answer in plain text."
            )
            messages = [Message(role="system", content=system_prompt)]
            messages.extend(state.chat_request.messages)
            agent_result = await llm_provider.invoke_tool_agent(messages, available_tools)
            if isinstance(agent_result, dict) and "error" in agent_result:
                task.status = "failed"
                task.result = agent_result
                return agent_result
            if not agent_result.tool_calls:
                state.mark_task_complete(task.task_id, result={"text_output": agent_result.text})
                continue
            task_results = []
            for tool_call in agent_result.tool_calls:
                execution_result = await tool_registry.execute(tool_call.name, tool_call.args)
                if isinstance(execution_result, dict) and "error" in execution_result:
                    task.status = "failed"
                    task.result = execution_result
                    return execution_result

                task_results.append({
                    "tool": tool_call.name,
                    "args": tool_call.args,
                    "output": execution_result
                })
            state.mark_task_complete(task.task_id, result=task_results)

        return {"status": "success", "message": "All planned tasks executed successfully."}

    except Exception as e:
        return {
            "message": "A critical error occurred in the execution loop.",
            "error": str(e),
            "status_code": 500
        }