import json
from typing import List, Dict, Any, AsyncGenerator
from .models import StateTracker, Message
from .providers.base import BaseLLMProvider
from .tools.registry import tool_registry
from .tools.artifact_service import export_dataset, process_odata_execution_result

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
            state.track_usage(agent_result)

            if agent_result.text:
                yield f"data: {json.dumps({'type': 'text', 'chunk': agent_result.text + '\n\n'})}\n\n"

            if not agent_result.tool_calls:
                state.mark_task_complete(task.task_id, result={"text_output": agent_result.text})
                continue
            task_results = []
            for tc in agent_result.tool_calls: # First loop to stream the tool calling straight to the ui for smoother streaming.
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc.name, 'args': tc.args})}\n\n"
            for tc in agent_result.tool_calls: # Second loop to do our manual maskings, tool diversion.
                if tc.name == "export_dataset":
                    execution_result = await export_dataset(file_id=tc.args.get("file_id"), export_type=tc.args.get("export_type"))
                    print(execution_result)
                else:
                    execution_result = await tool_registry.execute(tc.name, tc.args)
                llm_friendly_result = execution_result
                ui_stream_result = execution_result
                parsed_result = execution_result
                if isinstance(execution_result, str):
                    try:
                        parsed_result = json.loads(execution_result)
                    except Exception:
                        pass
                if isinstance(parsed_result, dict):
                    if "llm_summary" in parsed_result:
                        llm_friendly_result = parsed_result["llm_summary"]    
                    elif parsed_result.get("response_type") == "data" and parsed_result.get("status") != "error":
                        masked_result: dict = await process_odata_execution_result(execution_result=parsed_result)
                        ui_stream_result = masked_result
                        artifact_summaries = []
                        for artifact in masked_result.get("artifacts", []):
                            f_id = artifact.get('file_id')
                            query = artifact.get('odata_query', 'N/A')
                            file_url = f"http://127.0.0.1:8000/v1/files/{f_id}"
                            artifact_summaries.append(f"File ID: {f_id} (Query: {query}) - Download: {file_url}")     
                        artifacts_str = " | ".join(artifact_summaries)
                        llm_friendly_result = f"Data retrieved successfully. Tell the user they can download it. Do not show sample data. Artifacts available: {artifacts_str}"
                    elif parsed_result.get("response_type") == "file":
                        artifact_details = parsed_result.get("artifacts", [{}])
                        f_id = artifact_details[0].get('file_id')
                        file_url = f"http://127.0.0.1:8000/v1/files/{f_id}"
                        llm_friendly_result = f"The file has been generated in the desired format and already sent to the UI for download. Tell the user they can download it here: {file_url} and this is the file_id of file where data is stored: {f_id}"  
                    elif parsed_result.get("response_type") == "chart" or "plotly_data" in parsed_result:
                        llm_friendly_result = "Chart generated successfully and is already displayed on the user's screen. Acknowledge this, but do not output any chart data."
                yield f"data: {json.dumps({'type': 'tool_result', 'name': tc.name, 'result': ui_stream_result})}\n\n"
                task_results.append({"tool": tc.name,"args": tc.args,"output": llm_friendly_result})
            state.mark_task_complete(task.task_id, result=task_results)

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Execution loop failed', 'details': str(e)})}\n\n"