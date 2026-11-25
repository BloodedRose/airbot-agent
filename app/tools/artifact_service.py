import uuid, os, httpx
import pandas as pd
from typing import Literal
from pathlib import Path
from app.tools.database_repositry import save_artifact, get_artifact
from dotenv import load_dotenv
load_dotenv()

SANDBOX_URL = os.environ["SANDBOX_URL"]

MIME = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "xml": "application/xml",
    "md": "text/markdown",
    "json": "application/json",
    }

STORAGE_ROOT = Path(os.getenv("FILE_STORAGE_ROOT", "./files")).resolve()
MAX_INPUT_FILE_MB = int(os.getenv("MAX_INPUT_FILE_MB", "250"))
MAX_OUTPUT_ROWS = int(os.getenv("MAX_OUTPUT_ROWS", "1000000"))


async def write_dataframe(df: pd.DataFrame, extension="json", filename=None, source_type="odata", query_context_id=None):
    extension = extension.lower().lstrip(".")
    extensions = {
        "csv": "csv",
        "excel": "xlsx",
        "xlsx": "xlsx",
        "html": "html",
        "xml": "xml",
        "markdown": "md",
        "md": "md",
        "json": "json",
    }
    extension = extensions.get(extension, extension)
    if extension not in MIME:
        raise ValueError(f"Unsupported output format: {extension}")
    if len(df) > MAX_OUTPUT_ROWS:
        raise ValueError(f"Output exceeds MAX_OUTPUT_ROWS={MAX_OUTPUT_ROWS}")
    artifact_id = f"file_{uuid.uuid4().hex}"
    safe_name = Path(filename or f"{artifact_id}.{extension}").name
    if not safe_name.lower().endswith(f".{extension}"):
        safe_name += f".{extension}"
    path = STORAGE_ROOT / extension / f"{artifact_id}_{safe_name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if extension == "csv":
        df.to_csv(path, index=False)
    elif extension == "xlsx":
        df.to_excel(path, index=False)
    elif extension == "html":
        df.to_html(path, index=False, border=0)
    elif extension == "xml":
        df.to_xml(path, index=False)
    elif extension == "md":
        df.to_markdown(path, index=False)
    elif extension == "json":
        df.to_json(path,orient="records",date_format="iso",force_ascii=False,indent=2)
    schema = [{"column": str(column),"dtype": str(df[column].dtype),"null_count": int(df[column].isna().sum()),}for column in df.columns]
    artifact = await save_artifact(id=artifact_id, filename=safe_name, extension=extension, 
                content_type=MIME[extension], storage_path=str(path), size_bytes=path.stat().st_size, 
                row_count=len(df), column_count=len(df.columns), schema_json={"columns": schema}, 
                source_type=source_type, source_query_context_id=query_context_id)
    return {"file_id": artifact.id, "filename": artifact.filename, "extension": artifact.extension, 
            "content_type": artifact.content_type, "row_count": artifact.row_count, 
            "column_count": artifact.column_count, "size_bytes": artifact.size_bytes,
            }, (artifact.storage_path, artifact.filename)


def extract_odata_records(payload: dict | list) -> list:
    """
    Supports OData V4 and V2 response structures.
    """

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Invalid OData response payload")
    if isinstance(payload.get("value"), list): # OData V4
        return payload["value"]
    d = payload.get("d") # OData V2
    if isinstance(d, dict):
        if isinstance(d.get("results"), list):
            return d["results"]
        return [d] # Single-record OData V2 response
    raise ValueError("Unsupported OData response structure")


async def process_odata_execution_result(execution_result: dict) -> dict:
    artifacts = []
    errors = []
    results = execution_result.get("results", [])
    for result in results:
        if result.get("status") != "success":
            errors.append({"query_index": result.get("query_index"), "error": result.get("error", "SAP query execution failed")})
            continue

        try:
            service_name = result.get("service_name")
            entity_name = result.get("entity_name")
            raw_payload = result.get("data")
            if not service_name or not entity_name:
                raise ValueError("Missing service or entity name")
            records = extract_odata_records(raw_payload)
            dataframe = pd.DataFrame(records)
            filename = f"{service_name}_{entity_name}.json"
            artifact, _ = await write_dataframe(df=dataframe, extension="json", filename=filename, source_type="odata")
            artifacts.append({ "type": "file", "file_id": artifact["file_id"], "filename": artifact["filename"],
            "format": artifact["extension"], "mime_type": artifact["content_type"], 
            "row_count": artifact["row_count"], "column_count": artifact["column_count"], 
            "preview_type": "table", "odata_query": result.get("odata_query", "")})

        except Exception as e:
            errors.append({"query_index": result.get("query_index"), "service_name": result.get("service_name"), 
                           "entity_name": result.get("entity_name"), "error": str(e)})


    total_records = sum(
        artifact["row_count"]
        for artifact in artifacts
    )

    return {
        "status": ("success" if artifacts and not errors else "partial_success" if artifacts else "error"),
        "response_type": "file", 
        "message": (f"Successfully retrieved {total_records} records." if artifacts else "Failed to retrieve data."), 
        "artifacts": artifacts, 
        "errors": errors,}

async def load_dataframe(file_id: str) -> pd.DataFrame:
    artifact = await get_artifact(file_id)
    if not artifact:
        raise ValueError(f"Unknown file_id: {file_id}")
    path = Path(artifact.storage_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file is missing: {file_id}")
    if artifact.extension == "json":
        return pd.read_json(path, orient="records")
    if artifact.extension == "xlsx":
        return pd.read_excel(path)
    if artifact.extension == "csv":
        return pd.read_csv(path)
    if artifact.extension == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported stored extension: {artifact.extension}")

async def export_dataset(file_id: str, export_type: Literal["csv","excel","html","xml","markdown","jsonl"], tool_call:bool=True) -> dict:
    try:
        export_type = export_type.lower().strip().lstrip(".")
        extensions = {
            "csv": "csv",
            "excel": "xlsx",
            "xlsx": "xlsx",
            "html": "html",
            "xml": "xml",
            "markdown": "md",
            "md": "md",
            "json": "json"
        }
        if export_type not in extensions:
            return {"status": "error", "message": f"Unsupported export format: {export_type}"}
        extension = extensions[export_type]
        df = await load_dataframe(file_id=file_id)
        if df is None:
            return {"status": "error", "message": f"Unable to load dataset: {file_id}"}
        filename = f"export_{file_id}.{extension}"
        file_io, file_details = await write_dataframe(df=df, extension=extension, filename=filename, source_type="export")
        if tool_call:
            return {
                "status": "success", "response_type":"file",
                "message": f"Dataset successfully exported to {extension.upper()}",
                "data": file_io,
                "artifacts": [{"file_id": file_io["file_id"],"filename": file_io["filename"],"preview_type": "attachment"}]
                }
        else:
            return file_details
    except Exception as e:
        return {"status": "error", "message": f"Dataset export failed: {str(e)}"}


async def get_file_schema(file_id:str)-> dict:
    try:
        file_data = await get_artifact(file_id=file_id)
        return file_data.schema_json
    except Exception as e:
        return {"status": "error", "message": f"Schema fetch failed: {str(e)}"}

async def load_file_generate_chart(file_id:str, code:str)->dict:
    try:
        file_data = await get_artifact(file_id=file_id)
        sandbox_url = SANDBOX_URL
        async with httpx.AsyncClient() as client:
            with open(file_data.storage_path, "rb") as f:
                files = {"file": ("dataset.json", f, "application/json")}
                data = {"code": code}
                response = await client.post(sandbox_url, files=files, data=data, timeout=20.0)                
                return response.json()
    except Exception as e:
        return {"status": "error", "message": f"Chart generation failed: {str(e)}"}