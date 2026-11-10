from fastapi import FastAPI, Depends, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db, init_db
from .crud import get_model_config, create_model_config, create_mcp_config, get_mcp_configs, get_all_mcp_configs, get_all_model_configs
from .models import ChatRequest, StateTracker, ModelConfigCreate, MCPServerConfigCreate
from .providers.factory import get_llm_provider
from .planner import generate_plan
from .executor import execute_plan
from .synthesizer import stream_final_response
from .mcp_client import fetch_and_register_mcp_tools
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Helium")
render= Jinja2Templates(directory="app/templates").TemplateResponse


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#======================================== App Routes ========================================================


# ------- Configuration Endpoints -------

@app.get("/")
async def index(request:Request):
    await init_db()
    return render(request=request, name="chatbot.html")

@app.post("/v1/models/config")
async def add_model_config(config: ModelConfigCreate, db: AsyncSession = Depends(get_db)):
    result = await create_model_config(db, config.model_dump())
    
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(status_code=result.get("status_code", 400), content=result)
        
    return JSONResponse(status_code=201, content=result)


@app.post("/v1/mcp/config")
async def add_mcp_config(config: MCPServerConfigCreate, db: AsyncSession = Depends(get_db)):
    result = await create_mcp_config(db, config.model_dump())
    
    if isinstance(result, dict) and "error" in result:
        return JSONResponse(status_code=result.get("status_code", 400), content=result)
        
    return JSONResponse(status_code=201, content=result)


# ------- Main Orchestrator Endpoint -------

@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        db_config = await get_model_config(db, display_name=request.model_name)
        if isinstance(db_config, dict) and "error" in db_config:
            return JSONResponse(status_code=db_config.get("status_code", 400), content=db_config)
        mcp_configs = await get_mcp_configs(db, server_names=request.mcp_server_names)
        if isinstance(mcp_configs, dict) and "error" in mcp_configs:
            return JSONResponse(status_code=mcp_configs.get("status_code", 400), content=mcp_configs)  
        llm_provider = get_llm_provider(db_config)
        if isinstance(llm_provider, dict) and "error" in llm_provider:
            return JSONResponse(status_code=llm_provider.get("status_code", 400), content=llm_provider)
        state = StateTracker(chat_request=request)
        available_tools = await fetch_and_register_mcp_tools(mcp_configs)
        if isinstance(available_tools, dict) and "error" in available_tools:
            return JSONResponse(status_code=available_tools.get("status_code", 500), content=available_tools)
        plan_result = await generate_plan(state, llm_provider, available_tools)
        if isinstance(plan_result, dict) and "error" in plan_result:
            return JSONResponse(status_code=plan_result.get("status_code", 500), content=plan_result)
        print("THE AGENTIC PLAN:\n", plan_result)   
        execution_result = await execute_plan(state, llm_provider, available_tools)
        if isinstance(execution_result, dict) and "error" in execution_result:
            return JSONResponse(status_code=execution_result.get("status_code", 500), content=execution_result)
        print("FINAL_STATE:\n", state)  
        return StreamingResponse(stream_final_response(state, llm_provider),media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500,content={"message": "Critical server error in chat endpoint", "error": str(e),"status_code": 500})

# -------- Information Endpoints ---------

@app.get("/v1/models")
async def list_models(db: AsyncSession = Depends(get_db)):
    models = await get_all_model_configs(db)
    if isinstance(models, dict) and "error" in models:
        return JSONResponse(status_code=models.get("status_code", 500), content=models)
    safe_models = [
        {
            "display_name": m.display_name,
            "actual_name": m.actual_name,
            "provider": m.provider.value,
        }
        for m in models
    ]
    return JSONResponse(status_code=200, content={"status": "success", "models": safe_models})


@app.get("/v1/mcp/servers")
async def list_mcp_servers(db: AsyncSession = Depends(get_db)):
    servers = await get_all_mcp_configs(db)
    if isinstance(servers, dict) and "error" in servers:
        return JSONResponse(status_code=servers.get("status_code", 500), content=servers)
    safe_servers = [
        {
            "name": s.name,
            "transport": s.transport.value,
            "sse_url": s.sse_url
        }
        for s in servers
    ]
    return JSONResponse(status_code=200, content={"status": "success", "mcp_servers": safe_servers})


@app.get("/v1/tools")
async def list_available_tools(
    mcp_servers: str = Query(None, description="Comma-separated list of MCP server names"), 
    db: AsyncSession = Depends(get_db)
):
    try:
        if mcp_servers:
            server_names = [name.strip() for name in mcp_servers.split(",")]
            mcp_configs = await get_mcp_configs(db, server_names)
        else:
            mcp_configs = await get_all_mcp_configs(db)
        if isinstance(mcp_configs, dict) and "error" in mcp_configs:
            return JSONResponse(status_code=mcp_configs.get("status_code", 400), content=mcp_configs)
        available_tools = await fetch_and_register_mcp_tools(mcp_configs)
        return JSONResponse(status_code=200, content={"status": "success", "tools": available_tools})
        
    except Exception as e:
        return JSONResponse(status_code=500,content={"message": "Critical error while fetching available tools.", "error": str(e),"status_code": 500})