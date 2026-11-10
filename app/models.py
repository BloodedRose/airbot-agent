from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# --- Configuration Creation Schemas ---

class ModelConfigCreate(BaseModel):
    display_name: str = Field(..., description="Unique name called by the UI")
    actual_name: str = Field(..., description="Internal LLM name (e.g., gpt-4o)")
    provider: str = Field(..., description="ollama, openai, gemini, anthropic")
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    additional_headers: Optional[Dict[str, str]] = None
    is_active: bool = True

class MCPServerConfigCreate(BaseModel):
    name: str = Field(..., description="Unique name called by the UI")
    transport: str = Field(..., description="stdio or sse")
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    sse_url: Optional[str] = None
    is_active: bool = True


# --- Standardized Execution Results (Normalization Layer) ---

class AgentToolCall(BaseModel):
    id: Optional[str] = Field(default=None, description="Tool call ID (required by OpenAI/Anthropic, null for Gemini/Ollama)")
    name: str = Field(..., description="The exact function name to execute")
    args: Dict[str, Any] = Field(..., description="Parsed JSON arguments for the tool")

class AgentResult(BaseModel):
    text: Optional[str] = None
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    raw_response: Any = Field(default=None, description="The original SDK response object for debugging")


# --- Frontend Request Models ---

class Message(BaseModel):
    role: str = Field(..., description="'user', 'assistant', 'system', or 'tool'")
    content: Optional[str] = None
    tool_calls: Optional[List[AgentToolCall]] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    model_name: str = Field(..., description="The display name of the model")
    mcp_server_names: List[str] = Field(default=[], description="List of MCP server names to connect")
    
    stream: bool = Field(default=True)


# --- Internal Agent State Models ---

class Task(BaseModel):
    task_id: int
    description: str = Field(..., description="What the LLM planner decided needs to be done")
    status: str = Field(default="pending", description="'pending', 'in_progress', 'completed', 'failed'")
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None

class Plan(BaseModel):
    tasks: List[Task] = []

class StateTracker(BaseModel):
    chat_request: ChatRequest
    plan: Optional[Plan] = None
    execution_history: List[Task] = []
    
    def mark_task_complete(self, task_id: int, result: Any):
        for task in self.plan.tasks:
            if task.task_id == task_id:
                task.status = "completed"
                task.result = result
                self.execution_history.append(task)
                break