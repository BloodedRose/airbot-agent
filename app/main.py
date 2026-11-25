from fastapi import FastAPI, Depends, Request, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import get_db, init_db
from .crud import get_model_config, create_model_config, create_mcp_config, get_mcp_configs, get_all_mcp_configs, get_all_model_configs
from .models import ChatRequest, StateTracker, ModelConfigCreate, MCPServerConfigCreate, UserCreate, UserAccessUpdate, TableRequest, Message
from .schema import User, ModelConfig, MCPServerConfig
from .providers.factory import get_llm_provider
from .planner import generate_plan
from .executor import execute_plan
from .synthesizer import stream_final_response
from .mcp_client import fetch_and_register_mcp_tools
from fastapi.templating import Jinja2Templates
from .react import run_react_agent
from fastapi.security import OAuth2PasswordRequestForm
from .auth import verify_password, create_access_token, get_current_user
from .tools.database_repositry import get_artifact
from .tools.artifact_service import export_dataset
from .tools.helpers import RedisPython
import json, math, os


app = FastAPI(title="Helium")
render= Jinja2Templates(directory="app/templates").TemplateResponse
redis_client = RedisPython()

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
async def chat_endpoint(request: ChatRequest, background_tasks:BackgroundTasks, db: AsyncSession = Depends(get_db)):
    try:
        user_result = await db.execute(select(User).where(User.email == request.user_email, User.is_active == True))
        user = user_result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=401, content={"error": "Unauthorized user", "status_code": 401})
        if not user.is_admin:
            allowed_model_names = [m.display_name for m in user.allowed_models]
            if request.model_name not in allowed_model_names:
                return JSONResponse(status_code=403, content={"error": f"You do not have access to model '{request.model_name}'", "status_code": 403})
            allowed_mcp_names = [m.name for m in user.allowed_mcp_servers]
            for requested_mcp in request.mcp_server_names:
                if requested_mcp not in allowed_mcp_names:
                    return JSONResponse(status_code=403, content={"error": f"You do not have access to MCP server '{requested_mcp}'", "status_code": 403})
        db_config = await get_model_config(db, display_name=request.model_name)
        if isinstance(db_config, dict) and "error" in db_config:
            return JSONResponse(status_code=db_config.get("status_code", 400), content=db_config)
        mcp_configs = await get_mcp_configs(db, server_names=request.mcp_server_names)
        if isinstance(mcp_configs, dict) and "error" in mcp_configs:
            return JSONResponse(status_code=mcp_configs.get("status_code", 400), content=mcp_configs)  
        available_tools = await fetch_and_register_mcp_tools(mcp_configs)
        if isinstance(available_tools, dict) and "error" in available_tools:
            return JSONResponse(status_code=available_tools.get("status_code", 500), content=available_tools)
        async def execute_with_fallback(current_config, attempt=1):
            llm_provider = get_llm_provider(current_config)
            if isinstance(llm_provider, dict) and "error" in llm_provider:
                yield f"data: {json.dumps(llm_provider)}\n\n"
                return
            history_dicts = await redis_client.get_mapped_history(request.chat_id, turns=3)
            history = [Message(**msg) for msg in history_dicts]
            print("HISTORY:\n \n", history, "\n \n")
            history_length = len(history) 
            history.append(Message(role="user", content=request.user_text))
            state = StateTracker(chat_request=ChatRequest(user_text=request.user_text, chat_id=request.chat_id, model_name=request.model_name, 
                    user_email=request.user_email, mcp_server_names=request.mcp_server_names, 
                    planner=request.planner))
            state.chat_request_messages = history
            final_assistant_text = ""
            try:
                if not request.planner:
                    async for event in run_react_agent(state, llm_provider, available_tools):
                        yield event
                    # print("STATE:\n", state.chat_request_messages)
                else:
                    plan_result = await generate_plan(state, llm_provider, available_tools)
                    if isinstance(plan_result, dict) and "error" in plan_result:
                        yield f"data: {json.dumps(plan_result)}\n\n"
                        return
                    async for execution_event in execute_plan(state, llm_provider, available_tools):
                        yield execution_event
                    async for chunk in stream_final_response(state, llm_provider):
                        yield chunk
                        try:
                            data = json.loads(chunk.replace("data: ", "").strip())
                            if data.get("type") == "text":
                                final_assistant_text += data.get("chunk", "")
                        except Exception:
                            pass
                    state.chat_request_messages.append(Message(role="assistant", content=final_assistant_text))
                    yield f"data: {json.dumps({'type': 'done', 'input_tokens': state.total_input_tokens, 'output_tokens': state.total_output_tokens})}\n\n"
            except Exception as e:
                # if attempt == 1 and current_config.fallback_display_name:
                #     yield f"data: {json.dumps({'type': 'fallback', 'message': f'Provider failed, switching to {current_config.fallback_display_name}'})}\n\n"
                #     fallback_config = await get_model_config(db, display_name=current_config.fallback_display_name)
                #     if not (isinstance(fallback_config, dict) and "error" in fallback_config):
                #         async for event in execute_with_fallback(fallback_config, attempt=2):
                #             yield event
                #         return
                yield f"data: {json.dumps({'type': 'error', 'message': 'Execution failed', 'details': str(e)})}\n\n"
            new_turn_messages = state.chat_request_messages[history_length:]
            if new_turn_messages:
                background_tasks.add_task(redis_client.store_turn, chat_id=request.chat_id, new_messages=new_turn_messages)
        return StreamingResponse(execute_with_fallback(db_config), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(status_code=500,content={"message": "Critical server error in chat endpoint", "error": str(e),"status_code": 500})


@app.post("/v1/get_table_response/")
async def load_table_files(request:Request, payload:TableRequest):
    try:
        if payload.page < 1:
            raise ValueError("Page must be greater than or equal to 1")
        if payload.page_size < 1:
            raise ValueError("Page size must be greater than or equal to 1")
        data= await get_artifact(file_id=payload.file_id)
        file_path= data.storage_path
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        records = data if isinstance(data, list) else data.get("data", [])
        if not isinstance(records, list):
            raise ValueError("JSON table data must be a list")
        total_records = len(records)
        total_pages = math.ceil(total_records / payload.page_size)
        start_index = (payload.page - 1) * payload.page_size
        end_index = start_index + payload.page_size
        paginated_data = records[start_index:end_index]
        return JSONResponse(status_code=200, content={"data": paginated_data,
            "pagination": {
                "current_page": payload.page,
                "page_size": payload.page_size,
                "total_records": total_records,
                "total_pages": total_pages,
                "has_next": payload.page < total_pages,
                "has_previous": payload.page > 1
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":f"An error occured while fetching the file data, error: {e}"})



@app.get("/v1/files/{file_id}")
async def download_file(file_id: str, export: bool = Query(default=False), export_format: str = Query(default="json")):
    try:
        if export:
            if export_format is None:
                return JSONResponse(status_code=404, content={"message":"Export format is mandatory to export the fields"})
            file_details= await export_dataset(file_id=file_id, export_type=export_format, tool_call=False)
            if isinstance(file_details, dict) and  file_details["status"]=="error":
                return JSONResponse(status_code=500, content=file_details)
            file_path, file_name = file_details
        else:
            data= await get_artifact(file_id=file_id)
            file_path= data.storage_path
            file_name= data.filename
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "File not found or expired"})
        return FileResponse(path=file_path, filename=file_name, media_type="application/octet-stream", content_disposition_type="attachment")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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


#-------- Admin Endpoints ---------------

@app.get("/admin/")
async def render_admin(request:Request):
    return render(request=request, name="admin.html")


@app.post("/v1/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        return JSONResponse(status_code=401, content={"error": "Incorrect email or password"})
        
    access_token = create_access_token(data={"sub": user.email, "is_admin": user.is_admin})
    return {"access_token": access_token, "token_type": "bearer", "is_admin": user.is_admin, "name": user.name}



@app.post("/v1/admin/users")
async def create_user(user_data: UserCreate, db: AsyncSession = Depends(get_db), current_admin: User = Depends(get_current_user)):
    try:
        new_user = User(
            email=user_data.email,
            name=user_data.name,
            is_admin=user_data.is_admin
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return JSONResponse(status_code=201, content={"status": "success", "user_id": new_user.id})
    except Exception as e:
        await db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e), "status_code": 500})

@app.post("/v1/admin/users/{user_id}/access")
async def assign_user_access(user_id: int, access_data: UserAccessUpdate, db: AsyncSession = Depends(get_db), current_admin: User = Depends(get_current_user)):
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=404, content={"error": "User not found", "status_code": 404})
        models_query = await db.execute(select(ModelConfig).where(ModelConfig.id.in_(access_data.model_ids)))
        mcp_query = await db.execute(select(MCPServerConfig).where(MCPServerConfig.id.in_(access_data.mcp_server_ids)))
        user.allowed_models = models_query.scalars().all()
        user.allowed_mcp_servers = mcp_query.scalars().all()
        await db.commit()
        return JSONResponse(status_code=200, content={"status": "success", "message": "Access updated."})
    except Exception as e:
        await db.rollback()
        return JSONResponse(status_code=500, content={"error": str(e), "status_code": 500})