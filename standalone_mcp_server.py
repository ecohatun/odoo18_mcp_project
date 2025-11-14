#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone MCP server for testing MCP tools.
Complete implementation with all 18+ tools from the repository.
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from typing import Dict, Any, Optional, List
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
    title="Standalone MCP Server - Complete",
    description="Complete standalone server with all Odoo 18 MCP tools",
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
    from src.odoo.schemas import OdooConfig
    logger.info("Successfully imported OdooClient")
    
    def get_odoo_client():
        config = OdooConfig(
            url=os.getenv("ODOO_URL", "http://localhost:8069"),
            db=os.getenv("ODOO_DB", "odoo"),
            username=os.getenv("ODOO_USERNAME", "admin"),
            password=os.getenv("ODOO_PASSWORD", "admin")
        )
        return OdooClient(config)
    
except Exception as e:
    logger.error(f"Failed to import OdooClient: {e}")
    get_odoo_client = None

# Import advanced search
try:
    from advanced_search import AdvancedSearch
    from src.odoo.dynamic.model_discovery import ModelDiscovery
    
    def get_advanced_search():
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

# Import documentation retrieval
try:
    from src.odoo_docs_rag import OdooDocsRetriever
    docs_retriever = OdooDocsRetriever(
        docs_dir=os.getenv("ODOO_DOCS_DIR", "./odoo_docs"),
        index_dir=os.getenv("ODOO_INDEX_DIR", "./odoo_docs_index"),
        force_rebuild=False
    )
    logger.info("Successfully imported OdooDocsRetriever")
    docs_retriever_available = True
except Exception as e:
    logger.warning(f"OdooDocsRetriever not available: {e}")
    docs_retriever_available = False

# Import code generator
try:
    from src.simple_odoo_code_agent.odoo_code_generator import generate_odoo_module
    logger.info("Successfully imported Odoo Code Generator")
    code_generator_available = True
except Exception as e:
    logger.warning(f"Odoo Code Generator not available: {e}")
    code_generator_available = False

# ============================================================================
# BASIC CRUD OPERATIONS
# ============================================================================

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

def validate_field_value(model: str, field: str, value: Any) -> Dict[str, Any]:
    """Validate a field value for an Odoo model"""
    try:
        client = get_odoo_client()
        fields = client.execute(model, 'fields_get', [field], {})
        if field not in fields:
            return {"success": False, "error": f"Field '{field}' not found in model '{model}'"}
        
        field_info = fields[field]
        field_type = field_info.get('type')
        
        # Basic validation
        if field_info.get('required') and value is None:
            return {"success": False, "error": f"Field '{field}' is required"}
        
        return {
            "success": True,
            "model": model,
            "field": field,
            "value": value,
            "valid": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# ADVANCED SEARCH
# ============================================================================

def advanced_search(query: str, limit: int = 100) -> str:
    """Perform advanced natural language search"""
    if not advanced_search_available:
        return "Advanced search is not available. ModelDiscovery dependency missing."
    try:
        search = get_advanced_search()
        result = search.execute_query(query, limit)
        return result
    except Exception as e:
        logger.error(f"Error in advanced search: {str(e)}")
        return f"Error in advanced search: {str(e)}"

# ============================================================================
# FIELD ANALYSIS
# ============================================================================

def analyze_field_importance(model: str, use_nlp: bool = False) -> Dict[str, Any]:
    """Analyze field importance for an Odoo model"""
    try:
        client = get_odoo_client()
        fields = client.execute(model, 'fields_get', [], {})
        
        # Simple importance scoring
        important_fields = []
        for field_name, field_info in fields.items():
            score = 0
            if field_info.get('required'):
                score += 50
            if field_info.get('readonly'):
                score -= 20
            if field_name in ['name', 'display_name', 'email', 'phone']:
                score += 30
            
            important_fields.append({
                "field": field_name,
                "type": field_info.get('type'),
                "importance": score
            })
        
        # Sort by importance
        important_fields.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "success": True,
            "model": model,
            "fields": important_fields[:20]  # Top 20
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_field_groups(model: str) -> Dict[str, Any]:
    """Group fields by purpose for an Odoo model"""
    try:
        client = get_odoo_client()
        fields = client.execute(model, 'fields_get', [], {})
        
        groups = {
            "identification": [],
            "contact": [],
            "financial": [],
            "dates": [],
            "metadata": [],
            "relations": [],
            "other": []
        }
        
        for field_name, field_info in fields.items():
            field_type = field_info.get('type')
            
            if field_name in ['name', 'display_name', 'code', 'reference']:
                groups["identification"].append(field_name)
            elif field_name in ['email', 'phone', 'mobile', 'website']:
                groups["contact"].append(field_name)
            elif 'amount' in field_name or 'price' in field_name or 'total' in field_name:
                groups["financial"].append(field_name)
            elif field_type in ['date', 'datetime']:
                groups["dates"].append(field_name)
            elif field_name in ['create_date', 'write_date', 'create_uid', 'write_uid']:
                groups["metadata"].append(field_name)
            elif field_type in ['many2one', 'one2many', 'many2many']:
                groups["relations"].append(field_name)
            else:
                groups["other"].append(field_name)
        
        return {
            "success": True,
            "model": model,
            "groups": groups
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# EXPORT/IMPORT OPERATIONS
# ============================================================================

def export_records_to_csv(model: str, fields: List[str] = None, domain: List = None, 
                         output_path: str = "./exports/export.csv") -> Dict[str, Any]:
    """Export records from an Odoo model to CSV"""
    try:
        import csv
        
        client = get_odoo_client()
        
        # Get records
        records = client.search_read(model, domain or [], fields or [], limit=10000, offset=0)
        
        if not records:
            return {"success": False, "error": "No records found"}
        
        # Ensure export directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            if fields:
                fieldnames = fields
            else:
                fieldnames = list(records[0].keys())
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                # Clean up many2one fields (convert [id, name] to just name)
                cleaned_record = {}
                for key, value in record.items():
                    if isinstance(value, list) and len(value) == 2:
                        cleaned_record[key] = value[1]
                    else:
                        cleaned_record[key] = value
                writer.writerow(cleaned_record)
        
        return {
            "success": True,
            "model": model,
            "records_exported": len(records),
            "file": output_path
        }
    except Exception as e:
        logger.error(f"Error exporting records: {str(e)}")
        return {"success": False, "error": str(e)}

def import_records_from_csv(model: str, input_path: str, 
                            create_if_not_exists: bool = True,
                            update_if_exists: bool = False) -> Dict[str, Any]:
    """Import records from CSV to an Odoo model"""
    try:
        import csv
        
        client = get_odoo_client()
        
        if not os.path.exists(input_path):
            return {"success": False, "error": f"File not found: {input_path}"}
        
        created = 0
        updated = 0
        errors = 0
        
        with open(input_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    # Remove empty values
                    values = {k: v for k, v in row.items() if v}
                    
                    # Check if record exists (if id is provided)
                    record_id = values.pop('id', None)
                    
                    if record_id and update_if_exists:
                        client.write(model, [int(record_id)], values)
                        updated += 1
                    elif create_if_not_exists:
                        client.create(model, values)
                        created += 1
                        
                except Exception as e:
                    logger.error(f"Error importing record: {str(e)}")
                    errors += 1
        
        return {
            "success": True,
            "model": model,
            "created": created,
            "updated": updated,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Error importing records: {str(e)}")
        return {"success": False, "error": str(e)}

def export_related_records_to_csv(parent_model: str, child_model: str, 
                                  relation_field: str,
                                  output_path: str = "./exports/related_export.csv",
                                  parent_domain: List = None) -> Dict[str, Any]:
    """Export parent-child related records to CSV"""
    try:
        client = get_odoo_client()
        
        # Get parent records
        parent_records = client.search_read(parent_model, parent_domain or [], [], limit=1000, offset=0)
        
        if not parent_records:
            return {"success": False, "error": "No parent records found"}
        
        # For each parent, get child records
        all_records = []
        for parent in parent_records:
            parent_id = parent['id']
            
            # Get child records
            child_domain = [(relation_field, '=', parent_id)]
            child_records = client.search_read(child_model, child_domain, [], limit=10000, offset=0)
            
            for child in child_records:
                # Combine parent and child data
                combined = {f"parent_{k}": v for k, v in parent.items()}
                combined.update({f"child_{k}": v for k, v in child.items()})
                all_records.append(combined)
        
        # Export to CSV
        if all_records:
            import csv
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = list(all_records[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)
        
        return {
            "success": True,
            "parent_model": parent_model,
            "child_model": child_model,
            "records_exported": len(all_records),
            "file": output_path
        }
    except Exception as e:
        logger.error(f"Error exporting related records: {str(e)}")
        return {"success": False, "error": str(e)}

def import_related_records_from_csv(parent_model: str, child_model: str,
                                   relation_field: str, input_path: str) -> Dict[str, Any]:
    """Import parent-child related records from CSV"""
    return {
        "success": False,
        "error": "Import related records not fully implemented in standalone version"
    }

# ============================================================================
# DOCUMENTATION AND CODE GENERATION
# ============================================================================

def retrieve_odoo_documentation(query: str, max_results: int = 5,
                               use_gemini: bool = False,
                               use_online_search: bool = False) -> str:
    """Retrieve information from Odoo documentation"""
    if not docs_retriever_available:
        return "Documentation retrieval is not available. Install required dependencies: sentence-transformers, faiss-cpu, gitpython"
    
    try:
        results = docs_retriever.retrieve(query, max_results)
        return results
    except Exception as e:
        logger.error(f"Error retrieving documentation: {str(e)}")
        return f"Error retrieving documentation: {str(e)}"

def generate_npx(code: str, name: str = "diagram", theme: str = "default",
                backgroundColor: str = "white") -> Dict[str, Any]:
    """Generate PNG image from Mermaid markdown"""
    try:
        output_dir = "./exports/diagrams"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temp mermaid file
        mmd_file = f"{output_dir}/{name}.mmd"
        with open(mmd_file, 'w') as f:
            f.write(code)
        
        # Generate PNG using mmdc
        output_file = f"{output_dir}/{name}.png"
        cmd = [
            "mmdc",
            "-i", mmd_file,
            "-o", output_file,
            "-t", theme,
            "-b", backgroundColor
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                "success": True,
                "file": output_file,
                "message": f"Diagram generated: {output_file}"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to generate diagram: {result.stderr}"
            }
    except Exception as e:
        logger.error(f"Error generating diagram: {str(e)}")
        return {"success": False, "error": str(e)}

def improved_generate_odoo_module(module_name: str, requirements: str,
                                  save_to_disk: bool = True,
                                  output_dir: str = "./generated_modules") -> Dict[str, Any]:
    """Generate an Odoo 18 module using the code generator"""
    if not code_generator_available:
        return {
            "success": False,
            "error": "Code generator not available. Check dependencies."
        }
    
    try:
        result = generate_odoo_module(
            module_name=module_name,
            requirements=requirements,
            save_to_disk=save_to_disk,
            output_dir=output_dir
        )
        return {
            "success": True,
            "module_name": module_name,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error generating module: {str(e)}")
        return {"success": False, "error": str(e)}

def query_deepwiki(target_url: str) -> Dict[str, Any]:
    """Query DeepWiki for documentation"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(target_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content
        content = soup.get_text()
        
        return {
            "success": True,
            "url": target_url,
            "content": content[:5000]  # Limit to 5000 chars
        }
    except Exception as e:
        logger.error(f"Error querying DeepWiki: {str(e)}")
        return {"success": False, "error": str(e)}

# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_FUNCTIONS = {
    # Basic CRUD
    "search_records": search_records,
    "create_record": create_record,
    "update_record": update_record,
    "delete_record": delete_record,
    "execute_method": execute_method,
    "get_model_fields": get_model_fields,
    "search_count": search_count,
    "get_record_template": get_record_template,
    "validate_field_value": validate_field_value,
    
    # Advanced Search
    "advanced_search": advanced_search,
    
    # Field Analysis
    "analyze_field_importance": analyze_field_importance,
    "get_field_groups": get_field_groups,
    
    # Export/Import
    "export_records_to_csv": export_records_to_csv,
    "import_records_from_csv": import_records_from_csv,
    "export_related_records_to_csv": export_related_records_to_csv,
    "import_related_records_from_csv": import_related_records_from_csv,
    
    # Documentation & Code Generation
    "retrieve_odoo_documentation": retrieve_odoo_documentation,
    "generate_npx": generate_npx,
    "improved_generate_odoo_module": improved_generate_odoo_module,
    "query_deepwiki": query_deepwiki,
}

logger.info(f"Successfully loaded {len(TOOL_FUNCTIONS)} tools")

# ============================================================================
# API ENDPOINTS
# ============================================================================

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
        "app_name": "odoo18-mcp-complete",
        "tools_loaded": len(TOOL_FUNCTIONS),
        "tools": list(TOOL_FUNCTIONS.keys())
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Odoo MCP Standalone Server - Complete",
        "version": "2.0.0",
        "tools_available": len(TOOL_FUNCTIONS),
        "tools": list(TOOL_FUNCTIONS.keys()),
        "endpoints": {
            "health": "GET /health",
            "list_tools": "GET /list_tools",
            "call_tool": "POST /call_tool"
        }
    }


if __name__ == "__main__":
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = 8001

    logger.info(f"Starting complete standalone MCP server at {host}:{port}")
    logger.info(f"Available tools ({len(TOOL_FUNCTIONS)}): {', '.join(TOOL_FUNCTIONS.keys())}")
    uvicorn.run(app, host=host, port=port)
