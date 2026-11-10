import json
from typing import Dict, Any, List, Optional
from anthropic import AsyncAnthropic
from app.providers.base import BaseLLMProvider
from app.models import Message

class AnthropicAdapter(BaseLLMProvider):
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        super().__init__(model_name)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers
        )

    def _extract_system_and_messages(self, messages: List[Message]):
        """Anthropic expects system prompts separate from user/assistant messages."""
        system_prompt = ""
        formatted_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt += msg.content + "\n"
            else:
                formatted_messages.append({"role": msg.role, "content": msg.content})
        return system_prompt.strip(), formatted_messages

    async def generate_structured_json(self, messages: List[Message], schema: Dict[str, Any]) -> Any:
        try:
            system, formatted = self._extract_system_and_messages(messages)
            
            # Anthropic handles JSON schema best by forcing a tool call constraint
            tool = {
                "name": "output_json",
                "description": "Output the final result matching the schema",
                "input_schema": schema
            }
            
            response = await self.client.messages.create(
                model=self.model_name,
                system=system,
                messages=formatted,
                tools=[tool],
                tool_choice={"type": "tool", "name": "output_json"},
                max_tokens=4096,
                temperature=0.0
            )
            
            # Extract the arguments from the forced tool call
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            return {}

        except Exception as e:
            return {
                "message": f"An error happened while generating structured JSON from Anthropic model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def invoke_tool_agent(self, messages: List[Message], tools: List[Dict[str, Any]]) -> Any:
        try:
            system, formatted = self._extract_system_and_messages(messages)
            
            # Note: You will need a helper elsewhere to convert OpenAI-style tool schemas to Anthropic's 'input_schema'
            response = await self.client.messages.create(
                model=self.model_name,
                system=system,
                messages=formatted,
                tools=tools,
                max_tokens=4096
            )
            return response

        except Exception as e:
            return {
                "message": f"An error happened while invoking the tool agent in Anthropic model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def stream_text(self, messages: List[Message]):
        try:
            system, formatted = self._extract_system_and_messages(messages)
            async with self.client.messages.stream(
                model=self.model_name,
                system=system,
                messages=formatted,
                max_tokens=4096
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield {
                "message": f"An error happened while streaming text from Anthropic model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }