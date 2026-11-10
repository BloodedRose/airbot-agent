from typing import Callable, Dict, Any
import asyncio

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Registers a Python function as an executable tool."""
        self._tools[name] = func

    async def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """Executes the tool and catches standard errors."""
        if name not in self._tools:
            return {
                "message": f"Tool '{name}' is not registered in the system.",
                "error": "ToolNotFoundError",
                "status_code": 404
            }
            
        func = self._tools[name]
        try:
            # Handle both async and sync tools
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return result
            
        except Exception as e:
            return {
                "message": f"An error happened while executing the tool '{name}'.",
                "error": str(e),
                "status_code": 500
            }

# Global registry instance
tool_registry = ToolRegistry()

# --- Example Local Tool ---
def get_weather(location: str) -> dict:
    # Mock weather tool
    return {"location": location, "temperature": "32°C", "condition": "Sunny"}

tool_registry.register("get_weather", get_weather)