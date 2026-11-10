import json
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from app.providers.base import BaseLLMProvider
from app.models import Message, AgentResult, AgentToolCall

class OpenAIAdapter(BaseLLMProvider):
    def __init__(self, model_name: str, api_key: str, base_url: str = None, headers: dict = None):
        super().__init__(model_name)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers)

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            if msg.role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": json.dumps(msg.tool_result)
                })
            elif msg.role == "assistant" and msg.tool_calls:
                raw_calls = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)}
                } for tc in msg.tool_calls]
                formatted.append({"role": "assistant", "tool_calls": raw_calls, "content": msg.content})
            else:
                formatted.append({"role": msg.role, "content": msg.content})
        return formatted

    async def generate_structured_json(self, messages: List[Message], schema: Dict[str, Any]) -> Any:
        try:
            # OpenAI structured outputs require wrapping the schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True
                }
            }
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self._format_messages(messages),
                response_format=response_format,
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            return {
                "message": f"An error happened while generating structured JSON from OpenAI model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def invoke_tool_agent(self, messages: List[Message], tools: List[Dict[str, Any]]) -> AgentResult | dict:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self._format_messages(messages),
                tools=tools
            )
            
            msg = response.choices[0].message
            
            # --- NORMALIZATION LAYER ---
            normalized_tools = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    normalized_tools.append(AgentToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments)
                    ))

            return AgentResult(
                text=msg.content,
                tool_calls=normalized_tools,
                raw_response=response
            )

        except Exception as e:
            return {
                "message": f"An error happened while invoking the tool agent in OpenAI model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def stream_text(self, messages: List[Message]):
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self._format_messages(messages),
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield {
                "message": f"An error happened while streaming text from OpenAI model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }