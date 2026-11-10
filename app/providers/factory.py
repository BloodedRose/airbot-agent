from .base import BaseLLMProvider
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter
from app.schema import ModelConfig

def get_llm_provider(db_config: ModelConfig) -> BaseLLMProvider | dict:
    try:
        provider = db_config.provider.value
        actual_name = db_config.actual_name
        headers = db_config.additional_headers or {}
        
        if provider == "ollama":
            host = db_config.base_url or "http://localhost:11434"
            return OllamaAdapter(model_name=actual_name, host=host, headers=headers)
            
        elif provider == "openai":
            if not db_config.api_key:
                raise ValueError("OpenAI requires an API key in the database configuration.")
            return OpenAIAdapter(
                model_name=actual_name, 
                api_key=db_config.api_key, 
                base_url=db_config.base_url, 
                headers=headers
            )
            
        elif provider == "anthropic":
            if not db_config.api_key:
                raise ValueError("Anthropic requires an API key in the database configuration.")
            return AnthropicAdapter(
                model_name=actual_name, 
                api_key=db_config.api_key, 
                base_url=db_config.base_url,
                headers=headers
            )
            
        elif provider == "gemini":
            if not db_config.api_key:
                raise ValueError("Gemini requires an API key in the database configuration.")
            return GeminiAdapter(model_name=actual_name, api_key=db_config.api_key)
            
        else:
            return {
                "message": f"Provider '{provider}' configured for '{db_config.display_name}' is not supported.",
                "error": "UnsupportedProviderError",
                "status_code": 400
            }
            
    except Exception as e:
        return {
            "message": f"Failed to initialize provider from config '{db_config.display_name}'.",
            "error": str(e),
            "status_code": 500
        }