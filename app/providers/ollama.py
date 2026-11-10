import json
from typing import Dict, Any, List, Optional
from ollama import AsyncClient
from app.providers.base import BaseLLMProvider
from app.models import Message, AgentResult, AgentToolCall

class OllamaAdapter(BaseLLMProvider):
    def __init__(self, model_name: str, host: str = "http://localhost:11434", headers: dict = None):
        super().__init__(model_name)
        self.client = AsyncClient(host=host, headers=headers or {})

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            if msg.role == "tool":
                formatted.append({
                    "role": "tool",
                    "content": json.dumps(msg.tool_result)
                })
            elif msg.role == "assistant" and msg.tool_calls:
                raw_calls = [{
                    "function": {"name": tc.name, "arguments": tc.args}
                } for tc in msg.tool_calls]
                formatted.append({"role": "assistant", "tool_calls": raw_calls, "content": msg.content})
            else:
                formatted.append({"role": msg.role, "content": msg.content})
        return formatted
    
    async def generate_structured_json(self, messages: List[Message], schema: Dict[str, Any]) -> Any:
        try:
            formatted_messages = self._format_messages(messages)
            response = await self.client.chat(
                model=self.model_name,
                messages=formatted_messages,
                format=schema,
                options={"temperature": 0.0} # Low temperature for reliable planning
            )
            content = response.get("message", {}).get("content", "{}")
            return json.loads(content)

        except Exception as e:
            return {
                "message": f"An error happened while generating structured JSON from Ollama model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def invoke_tool_agent(self, messages: List[Message], tools: List[Dict[str, Any]]) -> AgentResult | dict:
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=self._format_messages(messages),
                tools=tools, 
                think=False
            )
            msg_obj = response.get("message", {})
            raw_tool_calls = msg_obj.get("tool_calls", [])
            normalized_calls = []
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    func = tc.get("function", {})
                    normalized_calls.append(AgentToolCall(
                        id=None,
                        name=func.get("name", ""),
                        args=func.get("arguments", {})
                    )) 
            return AgentResult(
                text=msg_obj.get("content"),
                tool_calls=normalized_calls,
                raw_response=response
            )
        except Exception as e:
            return {
                "message": f"An error happened while invoking the tool agent in Ollama model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def stream_text(self, messages: List[Message]):
        try:
            formatted_messages = self._format_messages(messages)
            stream = await self.client.chat(
                model=self.model_name,
                messages=formatted_messages,
                stream=True
            )
            async for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            yield {
                "message": f"An error happened while streaming text from Ollama model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }