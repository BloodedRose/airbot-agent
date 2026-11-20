from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.schema import Artifact, QueryContext, QueryExecution


async def save_artifact(**kwargs):
    async with AsyncSessionLocal() as db:
        obj = Artifact(**kwargs)
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return obj


async def get_artifact(file_id: str):
    async with AsyncSessionLocal() as db:
        return await db.get(Artifact, file_id)


async def save_query_context(**kwargs):
    async with AsyncSessionLocal() as db:
        obj = QueryContext(**kwargs)
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return obj


async def get_query_context(query_context_id: str):
    async with AsyncSessionLocal() as db:
        return await db.get(QueryContext, query_context_id)


async def update_query_context(query_context_id: str, query_plan: dict):
    async with AsyncSessionLocal() as db:
        obj = await db.get(QueryContext, query_context_id)

        if not obj:
            return None

        obj.query_plan = query_plan
        await db.commit()
        await db.refresh(obj)
        return obj


async def save_execution(**kwargs):
    async with AsyncSessionLocal() as db:
        obj = QueryExecution(**kwargs)
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return obj


async def get_execution(execution_id: str):
    async with AsyncSessionLocal() as db:
        return await db.get(QueryExecution, execution_id)