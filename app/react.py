import json, re
from typing import AsyncGenerator, List, Dict, Any
from .models import StateTracker, Message
from .providers.base import BaseLLMProvider
from .tools.registry import tool_registry
from .tools .artifact_service import process_odata_execution_result, export_dataset, get_file_schema, load_file_generate_chart

#Refusal Regex's
REFUSAL_REGEX = re.compile(
    r"\b(i\s+(cannot|can't|am\s+unable|am\s+not\s+able|don't\s+have|do\s+not\s+have)|unable\s+to|not\s+(directly\s+)?(possible|supported)|no\s+(tool|way|function)s?\s+(to|for|that))\b",
    re.IGNORECASE
)
REFUSAL_NUDGE = "You have tools available and have not used any of them. Do not refuse yet. Use the tools to fetch the data this needs, and then answer. If it truly cannot be done after trying, say what you tried and what is missing."

async def run_react_agent(state: StateTracker, llm_provider: BaseLLMProvider, available_tools: List[Dict[str, Any]], max_steps: int = 8) -> AsyncGenerator[str, None]:
    messages = state.chat_request_messages.copy()
    nudged = False
    if not hasattr(state, 'artifact_registry'):
        state.artifact_registry = {}
    for step in range(max_steps):
        execution_messages = messages.copy()
        if state.artifact_registry:
            registry_context = "Active Session Artifacts (You can use these file_ids in your tools):\n"
            for fid, details in state.artifact_registry.items():
                registry_context += f"- File ID: {fid} | Query/Type: {details.get('query', details.get('type', 'N/A'))}\n"
            execution_messages.insert(0, Message(role="system", content=registry_context))

        agent_result = await llm_provider.invoke_tool_agent(execution_messages, available_tools)
        if isinstance(agent_result, dict) and "error" in agent_result:
            yield f"data: {json.dumps(agent_result)}\n\n"
            return
        # print("React Result:\n", agent_result, "\n")
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
            for tc in agent_result.tool_calls: # First loop to stream the tool calling straight to the ui for smoother streaming.
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc.name, 'args': tc.args})}\n\n"
            messages.append(Message(role="assistant", tool_calls=agent_result.tool_calls))
            for tc in agent_result.tool_calls: # Second loop to do our manual maskings, tool diversion.
                if tc.name == "export_dataset":
                    execution_result = await export_dataset(file_id=tc.args.get("file_id"), export_type=tc.args.get("export_type"))
                elif tc.name =="get_file_schema":
                    execution_result = await get_file_schema(file_id=tc.args.get("file_id"))
                elif tc.name == "generate_plotly_chart":
                    execution_result = await load_file_generate_chart(file_id=tc.args.get("file_id"), code=tc.args.get("code"))
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
                            state.artifact_registry[f_id] = {"query": query, "filename": artifact.get("filename", "")}         
                        artifacts_str = " | ".join(artifact_summaries)
                        llm_friendly_result = f"Data retrieved successfully. Tell the user they can download it. Do not show sample data. Artifacts available: {artifacts_str}"
                    elif parsed_result.get("response_type") == "file" and parsed_result.get("status") != "error":
                        artifact_details = parsed_result.get("artifacts", [{}])
                        f_id = artifact_details[0].get('file_id')
                        file_url = f"http://127.0.0.1:8000/v1/files/{f_id}"
                        llm_friendly_result = f"The file has been generated in the desired format and already sent to the UI for download. Tell the user they can download it here: {file_url} and this is the file_id of file where data is stored: {f_id}"
                        state.artifact_registry[f_id] = {"type": "exported_file", "filename": artifact_details[0].get("filename", "")}
                    elif parsed_result.get("response_type") == "chart" or "plotly_data" in parsed_result:
                        llm_friendly_result = "Chart generated successfully and is already displayed on the user's screen. Acknowledge this, but do not output any chart data."

                yield f"data: {json.dumps({'type': 'tool_result', 'name': tc.name, 'result': ui_stream_result})}\n\n"
                messages.append(Message(
                    role="tool", 
                    tool_call_id=tc.id, 
                    tool_name=tc.name, 
                    tool_result=llm_friendly_result
                ))
            continue
        final_assistant_text = ""
        async for chunk in llm_provider.stream_text(execution_messages):
            if isinstance(chunk, dict) and "error" in chunk:
                yield f"data: {json.dumps(chunk)}\n\n"
                break
            final_assistant_text += chunk
            yield f"data: {json.dumps({'type': 'text', 'chunk': chunk})}\n\n"
        messages.append(Message(role="assistant", content=final_assistant_text))
        break

    EPHEMERAL_TOOLS = {"get_sap_services", "get_sap_entities", "get_sap_fields", "get_file_schema", "get_server_info"}
    pruned_messages = []
    for i, msg in enumerate(messages):
        if msg.role == "assistant" and getattr(msg, "tool_calls", None):
            keep_tools = []
            for tc in msg.tool_calls:
                tc_name = getattr(tc, "name", getattr(tc, "function", {}).get("name"))
                if tc_name in EPHEMERAL_TOOLS:
                    continue
                    
                failed = False
                for next_msg in messages[i+1:]:
                    if next_msg.role == "user" or (next_msg.role == "assistant" and getattr(next_msg, "tool_calls", None)):
                        break # Stop looking if we hit the next turn
                        
                    if next_msg.role == "tool" and getattr(next_msg, "tool_name", "") == tc_name:
                        # Convert to string to easily check for errors
                        res_str = str(getattr(next_msg, "tool_result", getattr(next_msg, "content", "")))
                        if "Traceback" in res_str or "'error'" in res_str or '"error"' in res_str:
                            failed = True
                        break
                
                if not failed:
                    keep_tools.append(tc)
                    
            if not keep_tools:
                # If all tools failed, only keep this message if the LLM also spoke actual text
                if getattr(msg, "content", None) and str(msg.content).strip():
                    new_msg = msg.model_copy() if hasattr(msg, "model_copy") else msg.copy()
                    new_msg.tool_calls = None
                    pruned_messages.append(new_msg)
            else:
                new_msg = msg.model_copy() if hasattr(msg, "model_copy") else msg.copy()
                new_msg.tool_calls = keep_tools
                pruned_messages.append(new_msg)
                
        elif msg.role == "tool":
            tool_name = getattr(msg, "tool_name", getattr(msg, "name", ""))
            res_str = str(getattr(msg, "tool_result", getattr(msg, "content", "")))
            
            if tool_name in EPHEMERAL_TOOLS:
                continue
            if "Traceback" in res_str or "'error'" in res_str or '"error"' in res_str:
                continue # Drop the failed traceback output
                
            pruned_messages.append(msg)
        else:
            pruned_messages.append(msg)

    # 4. Sync only the PRUNED state back for the Redis handler
    state.chat_request_messages = pruned_messages
    yield f"data: {json.dumps({'type': 'done', 'input_tokens': state.total_input_tokens, 'output_tokens': state.total_output_tokens})}\n\n"