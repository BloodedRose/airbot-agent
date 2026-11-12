from typing import List, Dict, Any
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from .schema import MCPServerConfig
from app.tools.registry import tool_registry  

async def _execute_remote_tool(sse_url: str, tool_name: str, arguments: Dict[str, Any], headers: Dict[str, str]) -> Any:
    """Connects to the MCP server, executes the specific tool, and extracts the text result."""
    try:
        async with streamable_http_client(sse_url, headers=headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name=tool_name, arguments=arguments)
                parsed_text = []
                for content in result.content:
                    if content.type == "text":
                        parsed_text.append(content.text)
                        
                return "\n".join(parsed_text)
    except Exception as e:
        return {
            "message": f"Failed to execute remote tool '{tool_name}'.",
            "error": str(e),
            "status_code": 500
        }

def _create_mcp_tool_wrapper(sse_url: str, tool_name: str, headers: Dict[str, str]):
    """Creates a dynamic Python function closure bound to this specific remote tool."""
    async def dynamic_tool(**kwargs):
        return await _execute_remote_tool(sse_url, tool_name, kwargs, headers=headers)
    return dynamic_tool


async def fetch_and_register_mcp_tools(mcp_configs: List[MCPServerConfig]) -> List[Dict[str, Any]]:
    """Connects to MCP servers, fetches their schemas, and registers them for execution."""
    available_tools = []
    
    for config in mcp_configs:
        if config.transport.value not in ["sse", "streamable-http"] or not config.sse_url:
            continue
            
        print(f"Connecting to MCP Server: {config.name} at {config.sse_url}")
        auth_headers = config.env or {}
        try:
            async with streamable_http_client(config.sse_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    print(f"Successfully loaded {len(tools_response.tools)} tools from {config.name}")
                    for tool in tools_response.tools:
                        available_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema
                            }
                        })
                        tool_registry.register(
                            name=tool.name, 
                            func=_create_mcp_tool_wrapper(config.sse_url, tool.name, headers=auth_headers)
                        )
                        
        except Exception as e:
            print(f"Failed to connect to MCP server {config.name}: {e}")
            return {
                    "message": f"Failed to fetch and register remote tool",
                    "error": str(e),
                    "status_code": 500
                    }
    return available_tools