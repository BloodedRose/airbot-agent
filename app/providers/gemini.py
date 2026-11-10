import json
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from app.providers.base import BaseLLMProvider
from app.models import Message, AgentToolCall, AgentResult

class GeminiAdapter(BaseLLMProvider):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = genai.Client(api_key=api_key)

    def _format_for_gemini(self, messages: List[Message]):
        system_instruction = ""
        contents = []
        
        for msg in messages:
            if msg.role == "system":
                system_instruction += (msg.content or "") + "\n"
            elif msg.role == "tool":
                part = types.Part.from_function_response(
                    name=msg.tool_name,
                    response={"result": msg.tool_result}
                )
                contents.append(types.Content(role="user", parts=[part]))
            elif msg.role == "assistant" and msg.tool_calls:
                parts = []
                for tc in msg.tool_calls:
                    parts.append(types.Part.from_function_call(name=tc.name, args=tc.args))
                contents.append(types.Content(role="model", parts=parts))
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content or "")]))
        return system_instruction.strip(), contents

    async def generate_structured_json(self, messages: List[Message], schema: Dict[str, Any]) -> Any:
        try:
            system, contents = self._format_for_gemini(messages)
            config = types.GenerateContentConfig(
                system_instruction=system if system else None,
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0.0
            )
            
            # The client is synchronous by default but can be wrapped/handled in async contexts
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            return json.loads(response.text)

        except Exception as e:
            return {
                "message": f"An error happened while generating structured JSON from Gemini model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def invoke_tool_agent(self, messages: List[Message], tools: List[Dict[str, Any]]) -> AgentResult | dict:
        try:
            system, contents = self._format_for_gemini(messages)
            config = types.GenerateContentConfig(
                system_instruction=system if system else None,
                tools=convert_tools(tools)
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            tool_calls = []
            text_content = ""
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        tool_calls.append(AgentToolCall(
                            id=None, # Gemini does not use explicit string IDs
                            name=part.function_call.name,
                            args=dict(part.function_call.args)
                        ))
                    elif part.text:
                        text_content += part.text

            return AgentResult(
                text=text_content.strip() if text_content else None,
                tool_calls=tool_calls,
                raw_response=response
            )

        except Exception as e:
            return {
                "message": f"An error happened while invoking the tool agent in Gemini model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

    async def stream_text(self, messages: List[Message]):
        try:
            system, contents = self._format_for_gemini(messages)
            
            config = types.GenerateContentConfig(
                system_instruction=system if system else None
            )
            
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield {
                "message": f"An error happened while streaming text from Gemini model {self.model_name}",
                "error": str(e),
                "status_code": 500
            }

def convert_tools(openai_tools):

    declarations = []

    for tool in openai_tools:

        function = tool["function"]

        declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function["description"],
                parameters_json_schema=function["parameters"]
            )
        )

    return [
        types.Tool(
            function_declarations=declarations
        )
    ]