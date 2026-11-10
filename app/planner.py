import json
from typing import List, Dict, Any
from .models import StateTracker, Message, Plan, Task
from .providers.base import BaseLLMProvider

PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "Sequential ID of the task, starting from 1."
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of what needs to be done in this step."
                    },
                    "tool_name": {
                        "type": ["string", "null"],
                        "description": "The exact name of the tool to use, or null if no tool is needed."
                    },
                    "tool_args": {
                        "type": ["object", "null"],
                        "description": "A JSON object representing the arguments to pass to the tool, or null."
                    }
                },
                "required": ["task_id", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["tasks"],
    "additionalProperties": False
}

def _build_planner_prompt(available_tools: List[Dict[str, Any]]) -> str:
    return f"""You are an expert AI orchestrator and planner. Your job is to analyze the user's request and break it down into a logical, sequential list of executable tasks.

            AVAILABLE TOOLS:
            {json.dumps(available_tools, indent=2) if available_tools else "No tools available."}

            RULES:
            1. Analyze the conversation history to understand the goal.
            2. Break the goal down into distinct, sequential steps.
            3. If a step requires a tool, specify the exact `tool_name` from the AVAILABLE TOOLS list and predict the required `tool_args`.
            4. If a step requires reasoning, calculation, or synthesis without a specific tool, set `tool_name` and `tool_args` to null.
            5. The final step should usually be synthesizing the gathered data into a final response.
            6. You MUST output strictly in the requested JSON format.
            """

async def generate_plan(state: StateTracker, llm_provider: BaseLLMProvider, available_tools: List[Dict[str, Any]]) -> Any:
    try:
        system_prompt = _build_planner_prompt(available_tools)
        messages = [
            Message(role="system", content=system_prompt)
        ]
        messages.extend(state.chat_request.messages)
        result = await llm_provider.generate_structured_json(
            messages=messages,
            schema=PLAN_JSON_SCHEMA
        )
        if isinstance(result, dict) and "error" in result:
            return result
        tasks = []
        for t in result.get("tasks", []):
            tasks.append(Task(
                task_id=t["task_id"],
                description=t["description"],
                tool_name=t.get("tool_name"),
                tool_args=t.get("tool_args"),
                status="pending"
            ))
        state.plan = Plan(tasks=tasks)
        return {"status": "success", "plan": state.plan.model_dump()}
    except Exception as e:
        return {
            "message": "Failed to generate execution plan.",
            "error": str(e),
            "status_code": 500
        }