from abc import ABC, abstractmethod
from typing import Dict, Any, List
from app.models import Message, AgentResult

class BaseLLMProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def generate_structured_json(self, messages: List[Message], schema: Dict[str, Any]) -> Any:
        """
        Forces the LLM to output JSON matching a specific schema.
        Must return the parsed JSON dictionary, OR your standard error dictionary if it fails.
        """
        pass

    @abstractmethod
    async def invoke_tool_agent(self, messages: List[Message], tools: List[Dict[str, Any]]) -> AgentResult | dict:
        """
        Must return the standardized AgentResult containing text and tool_calls, 
        or the standard error dictionary if the API call fails.
        """
        pass

    @abstractmethod
    async def stream_text(self, messages: List[Message]):
        """
        Yields text chunks for Server-Sent Events (SSE).
        """
        pass