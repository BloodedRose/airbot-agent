from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .schema import ModelConfig, MCPServerConfig, ProviderEnum, TransportEnum
from typing import Dict, Any, List


async def get_model_config(db: AsyncSession, display_name: str) -> ModelConfig | Dict[str, Any]:
    try:
        query = select(ModelConfig).where(ModelConfig.display_name == display_name, ModelConfig.is_active == True)
        result = await db.execute(query)
        model_config = result.scalar_one_or_none()
        if not model_config:
            return {
                "message": f"Model configuration for '{display_name}' not found or is inactive.",
                "error": "ModelNotFoundError",
                "status_code": 404
            }
        return model_config
    except Exception as e:
        return {
            "message": f"An error happened while querying the database for model '{display_name}'",
            "error": str(e),
            "status_code": 500
        }



async def create_model_config(db: AsyncSession, config_data: dict) -> Dict[str, Any]:
    try:
        provider_enum = ProviderEnum(config_data['provider'].lower())
        new_model = ModelConfig(
            display_name=config_data['display_name'],
            actual_name=config_data['actual_name'],
            provider=provider_enum,
            base_url=config_data.get('base_url'),
            api_key=config_data.get('api_key'),
            additional_headers=config_data.get('additional_headers'),
            is_active=config_data.get('is_active', True)
        )
        db.add(new_model)
        await db.commit()
        await db.refresh(new_model)
        
        return {"status": "success", "id": new_model.id, "display_name": new_model.display_name}
    except Exception as e:
        await db.rollback()
        return {
            "message": "Failed to create model configuration in database.",
            "error": str(e),
            "status_code": 500
        }

async def create_mcp_config(db: AsyncSession, config_data: dict) -> Dict[str, Any]:
    try:
        transport_enum = TransportEnum(config_data['transport'].lower())
        new_mcp = MCPServerConfig(
            name=config_data['name'],
            transport=transport_enum,
            command=config_data.get('command'),
            args=config_data.get('args'),
            env=config_data.get('env'),
            sse_url=config_data.get('sse_url'),
            is_active=config_data.get('is_active', True)
        )
        db.add(new_mcp)
        await db.commit()
        await db.refresh(new_mcp)
        
        return {"status": "success", "id": new_mcp.id, "name": new_mcp.name}
    except Exception as e:
        await db.rollback()
        return {
            "message": "Failed to create MCP server configuration in database.",
            "error": str(e),
            "status_code": 500
        }

async def get_mcp_configs(db: AsyncSession, server_names: List[str]) -> Any:
    try:
        if not server_names:
            return []
            
        query = select(MCPServerConfig).where(
            MCPServerConfig.name.in_(server_names),
            MCPServerConfig.is_active == True
        )
        result = await db.execute(query)
        configs = result.scalars().all()
        found_names = {c.name for c in configs}
        missing = set(server_names) - found_names
        if missing:
            return {
                "message": f"The following MCP servers were not found or are inactive: {', '.join(missing)}",
                "error": "MCPServerNotFoundError",
                "status_code": 404
            }
            
        return configs
    except Exception as e:
        return {
            "message": "An error happened while querying the database for MCP servers.",
            "error": str(e),
            "status_code": 500
        }

async def get_all_model_configs(db: AsyncSession) -> Any:
    try:
        query = select(ModelConfig).where(ModelConfig.is_active == True)
        result = await db.execute(query)
        configs = result.scalars().all()
        return configs
    except Exception as e:
        return {
            "message": "Failed to fetch model configurations from database.",
            "error": str(e),
            "status_code": 500
        }

async def get_all_mcp_configs(db: AsyncSession) -> Any:
    try:
        query = select(MCPServerConfig).where(MCPServerConfig.is_active == True)
        result = await db.execute(query)
        configs = result.scalars().all()
        return configs
    except Exception as e:
        return {
            "message": "Failed to fetch MCP server configurations from database.",
            "error": str(e),
            "status_code": 500
        }