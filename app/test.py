from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
import httpx
# Create MCP server
mcp = MCPServer("python-demo-server")


@mcp.tool()
def get_server_info() -> dict:
    """Get information about this MCP server."""
    return {
        "name": "Python Demo MCP Server",
        "version": "1.0.0",
        "language": "Python",
        "status": "running"
    }


BASE_URL = "http://122.166.127.223:100"
PROJECT_ID = 6



@mcp.tool()
async def get_sap_services() -> dict:
    """
    Discover all available SAP OData services and their descriptions.

    Use this tool FIRST when a user asks a question that requires
    retrieving information from SAP.

    Your task is to identify the service that is most relevant to
    the user's business question by examining the service descriptions.

    IMPORTANT:
    - Do not invent service names.
    - Do not assume that a business term mentioned by the user is a service name.
    - Select services only from the returned service catalogue.
    - Use the exact service name returned by this tool in subsequent calls.

    Returns:
        Available SAP service names and their descriptions.
    """
    url = f"{BASE_URL}/api/get_service_desc"
    payload = {"project_id": PROJECT_ID}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_sap_entities(
    service_names: list[str]
) -> dict:
    """
    Retrieve entities and their descriptions for the specified SAP services.

    Use this tool AFTER get_sap_services.

    Args:
        service_names:
            A list of exact SAP service names obtained from get_sap_services.

    IMPORTANT:
    - Never pass business terms or entity names as service names.
    - Never invent service names.
    - Examine the entity descriptions to identify the entity that
      contains the information required by the user's question.
    - Use only the exact entity names returned by this tool
      in subsequent get_sap_fields and execute_odata_query calls.

    Returns:
        Entity names and their descriptions grouped by SAP service.
    """

    url = f"{BASE_URL}/api/get_entity_desc"

    payload = {
        "project_id": PROJECT_ID,
        "service_names": service_names
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_sap_fields(
    service_name: str,
    entity_names: list[str]
) -> dict:
    """
    Retrieve available fields and their metadata for SAP entities.

    Use this tool AFTER get_sap_entities.

    Args:
        service_name:
            Exact SAP service name obtained from get_sap_services.

        entity_names:
            List of exact entity names obtained from get_sap_entities
            for the specified service.

    IMPORTANT:
    - Do not invent service names or entity names.
    - Examine the returned field names and descriptions carefully.
    - Use only the exact field names returned by this tool when
      constructing filters and select parameters.
    - Do not use SAP labels as field names unless they match
      the actual field name.
    - Do not assume a field exists just because it is common in SAP.

    Returns:
        Field names, data types, labels and other available metadata.
    """

    url = f"{BASE_URL}/api/get_field_details"

    payload = {
        "project_id": PROJECT_ID,
        "service_name": service_name,
        "entity_names": entity_names
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def execute_odata_query(
    service_name: str,
    entity_name: str,
    filters: list[str] | None = None,
    select: list[str] | None = None,
) -> dict:
    """
Execute an OData query against an SAP service.

Use this tool ONLY AFTER discovering the service, entity and fields
using the other SAP discovery tools.

Args:
    service_name:
        Exact service name obtained from get_sap_services.

    entity_name:
        Exact entity name obtained from get_sap_entities.

    filters:
        List of OData filter expressions.
        Every field referenced in a filter MUST exist in the
        metadata returned by get_sap_fields.
        Multiple filter expressions are combined using AND.
        Use an empty list when no filtering is required.

    select:
        List of exact field names to retrieve.
        Every field MUST exist in the metadata returned by
        get_sap_fields.
        Use an empty list when all fields are required.

IMPORTANT:
- Never execute a query using invented service names.
- Never execute a query using invented entity names.
- Never invent field names in filters or select.
- Do not create unnecessary filters.
- If the user asks for all records, do not add a filter.
- If the user requests specific information, select only the
  relevant fields when possible.
- If a requested field or filter cannot be verified using the
  discovered metadata, do not execute the query.

Returns:
    Query execution status, query parameters and retrieved records.
"""

    # Dummy data for testing model tool-calling quality
    dummy_users = [
        {
            "InternetUser": "john@example.com",
            "Name": "John Doe",
            "Department": "IT",
            "SapUserid": "USER001",
            "CreatedBy": "ADMIN",
            "Mobile": "9876543210",
            "CreatedOn": "2026-01-15T10:30:00",
        },
        {
            "InternetUser": "alice@example.com",
            "Name": "Alice Smith",
            "Department": "Finance",
            "SapUserid": "USER002",
            "CreatedBy": "ADMIN",
            "Mobile": "9876543211",
            "CreatedOn": "2026-02-20T11:45:00",
        },
        {
            "InternetUser": "bob@example.com",
            "Name": "Bob Wilson",
            "Department": "IT",
            "SapUserid": "USER003",
            "CreatedBy": "MANAGER",
            "Mobile": "9876543212",
            "CreatedOn": "2026-03-10T09:15:00",
        },
    ]

    # Apply the requested select fields to the dummy response
    if select:
        records = [
            {field: record[field] for field in select if field in record}
            for record in dummy_users
        ]
    else:
        records = dummy_users

    return {
        "status": "success",
        "execution_mode": "dummy",
        "service_name": service_name,
        "entity_name": entity_name,
        "filters": filters or [],
        "select": select or [],
        "total_records": len(records),
        "data": records,
    }



if __name__ == "__main__":

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost",
            "localhost:*",
            "127.0.0.1",
            "127.0.0.1:*",

            # Replace with your actual public IP
            "122.166.127.223",
            "122.166.127.223:*",

            # Replace with your actual domain
            "mcp.yourdomain.com",
            "mcp.yourdomain.com:*",
        ]
    )

    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        transport_security=security
    )
