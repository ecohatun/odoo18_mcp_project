#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone MCP server for testing MCP tools.
This version implements tools directly without depending on mcp_server.py
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("standalone_mcp_server")

# Create FastAPI app
app = FastAPI(
    title="Standalone MCP Server",
    description="A standalone server for testing MCP tools",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import Odoo client and tools
try:
    from src.odoo.client import OdooClient
    logger.info("Successfully imported OdooClient")
    
    def get_odoo_client():
        return OdooClient(
            url=os.getenv("ODOO_URL", "http://localhost:8069"),
            db=os.getenv("ODOO_DB", "odoo"),
            username=os.getenv("ODOO_USERNAME", "admin"),
            password=os.getenv("ODOO_PASSWORD", "admin")
        )
    
except Exception as e:
    logger.error(f"Failed to import OdooClient: {e}")
    get_odoo_client = None

# Import advanced search
try:
    from advanced_search import AdvancedSearch
    from src.odoo.dynamic.model_discovery import ModelDiscovery
    
    def get_advanced_search():
        client = get_odoo_client()
        model_discovery = ModelDiscovery(
            url=os.getenv("ODOO_URL"),
            db=os.getenv("ODOO_DB"),
            username=os.getenv("ODOO_USERNAME"),
            password=os.getenv("ODOO_PASSWORD")
        )
        return AdvancedSearch(model_discovery)
    
    logger.info("Successfully imported AdvancedSearch")
    advanced_search_available = True
except Exception as e:
    logger.warning(f"AdvancedSearch not available: {e}")
    advanced_search_available = False

# Tool implementations
def search_records(model: str, domain: list = None, fields: list = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """Search records in Odoo"""
    try:
        client = get_odoo_client()
        records = client.search_read(model, domain or [], fields or [], limit, offset)
        return {
            "success": True,
            "model": model,
            "count": len(records),
            "records": records
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_record(model: str, values: dict) -> Dict[str, Any]:
    """Create a new record in Odoo"""
    try:
        client = get_odoo_client()
        record_id = client.create(model, values)
        return {
            "success": True,
            "model": model,
            "id": record_id,
            "message": f"Record created successfully with ID {record_id}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def update_record(model: str, id: int, values: dict) -> Dict[str, Any]:
    """Update an existing record in Odoo"""
    try:
        client = get_odoo_client()
        success = client.write(model, [id], values)
        return {
            "success": success,
            "model": model,
            "id": id,
            "message": f"Record {id} updated successfully" if success else "Update failed"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def delete_record(model: str, id: int) -> Dict[str, Any]:
    """Delete a record from Odoo"""
    try:
        client = get_odoo_client()
        success = client.unlink(model, [id])
        return {
            "success": success,
            "model": model,
            "id": id,
            "message": f"Record {id} deleted successfully" if success else "Delete failed"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_method(model: str, method: str, args: list = None, kwargs: dict = None) -> Dict[str, Any]:
    """Execute a custom method on an Odoo model"""
    try:
        client = get_odoo_client()
        result = client.execute(model, method, args or [], kwargs or {})
        return {
            "success": True,
            "model": model,
            "method": method,
            "result": result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_model_fields(model: str) -> Dict[str, Any]:
    """Get field definitions for an Odoo model"""
    try:
        client = get_odoo_client()
        fields = client.execute(model, 'fields_get', [], {})
        return {
            "success": True,
            "model": model,
            "fields": fields
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_count(model: str, domain: list = None) -> Dict[str, Any]:
    """Count records matching a domain"""
    try:
        client = get_odoo_client()
        count = client.execute(model, 'search_count', [domain or []], {})
        return {
            "success": True,
            "model": model,
            "count": count
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def advanced_search(query: str, limit: int = 100) -> str:
    """Perform advanced natural language search"""
    if not advanced_search_available:
        return "Advanced search is not available"
    try:
        search = get_advanced_search()
        result = search.execute_query(query, limit)
        return result
    except Exception as e:
        return f"Error in advanced search: {str(e)}"

def get_record_template(model: str) -> Dict[str, Any]:
    """Get a template for creating a new record"""
    try:
        client = get_odoo_client()
        fields = client.execute(model, 'fields_get', [], {})
        template = {}
        for field_name, field_info in fields.items():
            if field_info.get('required'):
                template[field_name] = None
        return {
            "success": True,
            "model": model,
            "template": template
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Tool registry
TOOL_FUNCTIONS = {
    "search_records": search_records,
    "create_record": create_record,
    "update_record": update_record,
    "delete_record": delete_record,
    "execute_method": execute_method,
    "get_model_fields": get_model_fields,
    "search_count": search_count,
    "advanced_search": advanced_search,
    "get_record_template": get_record_template,
}

logger.info(f"Successfully loaded {len(TOOL_FUNCTIONS)} tools")


@app.post("/call_tool")
async def call_tool(request: Request):
    """Call an MCP tool."""
    try:
        body = await request.json()
        
        tool_name = body.get("tool") or body.get("name")
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        arguments = body.get("params") or body.get("arguments") or {}
        
        if tool_name not in TOOL_FUNCTIONS:
            available = ", ".join(TOOL_FUNCTIONS.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Tool '{tool_name}' not found. Available: {available}"
            )
        
        tool_function = TOOL_FUNCTIONS[tool_name]
        
        logger.info(f"Calling tool '{tool_name}' with arguments: {arguments}")
        
        import inspect
        if inspect.iscoroutinefunction(tool_function):
            result = await tool_function(**arguments)
        else:
            result = tool_function(**arguments)
        
        return {
            "success": True,
            "tool": tool_name,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling tool: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/list_tools")
async def list_tools():
    """List all available tools."""
    tool_info = {}
    for tool_name, tool_func in TOOL_FUNCTIONS.items():
        tool_info[tool_name] = {
            "description": tool_func.__doc__ or f"Execute {tool_name}",
            "parameters": {}
        }

    return {"success": True, "tools": tool_info}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "tools_loaded": len(TOOL_FUNCTIONS),
        "tools": list(TOOL_FUNCTIONS.keys())
    }


if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = 8001

    logger.info(f"Starting standalone MCP server at {host}:{port}")
    logger.info(f"Available tools: {', '.join(TOOL_FUNCTIONS.keys())}")
    uvicorn.run(app, host=host, port=port)
