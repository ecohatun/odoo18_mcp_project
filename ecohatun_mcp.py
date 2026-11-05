"""Proposed Odoo 18 read-only MCP entry point for Ecohatun staging."""
import json
import logging
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date
from hmac import compare_digest
from typing import Literal
from urllib.parse import urlsplit
from xmlrpc.client import SafeTransport, ServerProxy, Transport

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.responses import JSONResponse
from typing_extensions import Annotated

IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")
MODEL = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*")
ROLES = {"comercial", "interno", "tecnico"}
OPERATORS = {"=", "!=", "<", "<=", ">", ">=", "in", "not in", "ilike"}
MAX_FIELDS = 24
MAX_ROWS = 100


@dataclass(frozen=True)
class Profile:
    fields: dict[str, frozenset[str]]
    token: bytes = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)


principal: ContextVar[Profile] = ContextVar("principal")
mcp = FastMCP("Ecohatun Odoo", stateless_http=True, json_response=True)


def configured_profiles():
    data = json.loads(os.environ["ODOO_MCP_MODEL_FIELDS"])
    if not isinstance(data, dict) or not data or set(data) - ROLES:
        raise ValueError("Invalid Odoo MCP profiles")
    profiles = []
    for role, models in data.items():
        if not isinstance(models, dict) or not models:
            raise ValueError("Missing model policy")
        allowed = {}
        for model, fields in models.items():
            if not MODEL.fullmatch(model) or not isinstance(fields, list) or not fields:
                raise ValueError("Invalid model policy")
            if len(fields) > MAX_FIELDS or any(
                not isinstance(name, str) or not IDENTIFIER.fullmatch(name)
                for name in fields
            ):
                raise ValueError("Invalid field policy")
            allowed[model] = frozenset(fields)
        prefix = f"ODOO_MCP_{role.upper()}"
        raw_token = os.environ[f"{prefix}_TOKEN"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", raw_token):
            raise ValueError("Invalid service token")
        token = raw_token.encode("ascii")
        username = os.environ[f"{prefix}_USER"].strip()
        password = os.environ[f"{prefix}_API_KEY"]
        if len(token) < 32 or not username or not password or username.lower() == "admin":
            raise ValueError("Invalid service identity")
        if any(compare_digest(token, p.token) or username == p.username for p in profiles):
            raise ValueError("Profiles require independent identities")
        profiles.append(Profile(allowed, token, username, password))
    return profiles


def checked_domain(domain, fields):
    # Restrict v1 to simple AND filters, with no relationship traversals.
    if len(domain) > 12:
        raise ValueError("Too many filters")
    for item in domain:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError("Unsupported filter")
        name, operator, value = item
        if not isinstance(name, str) or name not in fields:
            raise ValueError("Field not permitted")
        if not isinstance(operator, str) or operator not in OPERATORS:
            raise ValueError("Operator not permitted")
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list) != (operator in {"in", "not in"}) or len(values) > 50:
            raise ValueError("Unsupported filter value")
        if any(type(v) not in (str, int, float, bool) or isinstance(v, str) and len(v) > 256 for v in values):
            raise ValueError("Unsupported filter value")
    return domain


def model_fields(model):
    fields = principal.get().fields.get(model)
    if fields is None:
        raise ValueError("Model not permitted")
    return fields


class TimedConnection:
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 8
        return connection


class TimedHTTP(TimedConnection, Transport):
    pass


class TimedHTTPS(TimedConnection, SafeTransport):
    pass


def odoo_read(model, method, args, kwargs):
    from src.odoo.client import OdooClient
    from src.odoo.schemas import OdooConfig

    class ReadClient(OdooClient):
        def _setup_connection(self):
            transport = TimedHTTPS if urlsplit(self.config.url).scheme == "https" else TimedHTTP
            self.common = ServerProxy(self.config.url + "/xmlrpc/2/common", transport=transport())
            self.models = ServerProxy(self.config.url + "/xmlrpc/2/object", transport=transport())

    profile = principal.get()
    url = os.environ["ODOO_URL"].rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(
        (parsed.username, parsed.password, parsed.query, parsed.fragment)
    ):
        raise ValueError("Invalid Odoo endpoint")
    config = OdooConfig(
        url=url, db=os.environ["ODOO_DB"], username=profile.username, password=profile.password
    )
    try:
        # The inherited client logs remote exception messages and arguments.
        logging.getLogger("src.odoo.client").disabled = True
        return ReadClient(config).execute(model, method, args, kwargs)
    except Exception:
        raise ValueError("Odoo read failed") from None


Limit = Annotated[int, Field(ge=1, le=MAX_ROWS)]


@mcp.tool()
def list_allowed_models() -> list[str]:
    """List model names enabled for the authenticated profile."""
    return sorted(principal.get().fields)


@mcp.tool()
def describe_model(model: str) -> dict:
    """Read Odoo labels and types for fields enabled for this profile."""
    allowed = model_fields(model)
    metadata = odoo_read(model, "fields_get", [sorted(allowed)], {
        "attributes": ["string", "type"]
    })
    return {name: {k: value[k] for k in ("string", "type") if k in value}
            for name, value in metadata.items() if name in allowed}


@mcp.tool()
def search_records(model: str, fields: list[str], domain: list[list] | None = None, limit: Limit = 20) -> list[dict]:
    """Read explicitly authorized Odoo fields using bounded equality/range filters."""
    allowed = model_fields(model)
    if not fields or len(fields) > MAX_FIELDS or set(fields) - allowed:
        raise ValueError("Fields not permitted")
    records = odoo_read(model, "search_read", [checked_domain(domain or [], allowed)], {
        "fields": fields, "limit": limit
    })
    return [{k: row[k] for k in fields if k in row} for row in records]


@mcp.tool()
def search_count(model: str, domain: list[list] | None = None) -> int:
    """Count Odoo records visible to this profile."""
    return odoo_read(model, "search_count", [checked_domain(domain or [], model_fields(model))], {})


@mcp.tool()
def aggregate_records(
    model: str, group_by: str, sum_field: str | None = None,
    domain: list[list] | None = None, limit: Limit = 20
) -> list[dict]:
    """Compute grouped counts and an optional sum in Odoo 18 ORM."""
    allowed = model_fields(model)
    if group_by not in allowed or (sum_field is not None and (sum_field not in allowed or sum_field == group_by)):
        raise ValueError("Group or measure not permitted")
    # Named aggregate syntax gives a stable output field.
    measures = [group_by, "record_count:count(id)"]
    if sum_field:
        measures.append(f"total:sum({sum_field})")
    rows = odoo_read(model, "read_group", [checked_domain(domain or [], allowed), measures, [group_by]], {
        "limit": limit, "lazy": False
    })
    fields = {group_by, "record_count", "total"}
    return [{k: row[k] for k in fields if k in row} for row in rows]


SALES_STATUS_QUERY = "sale_orders_by_status.v1"


@mcp.tool()
def list_saved_queries() -> list[dict]:
    """List versioned query templates available to this profile."""
    fields = principal.get().fields.get("sale.order", frozenset())
    if {"state", "date_order"} <= fields:
        return [{"id": SALES_STATUS_QUERY, "description": "Sales order counts by status over a UTC date range"}]
    return []


@mcp.tool()
def run_saved_query(
    query_id: Literal["sale_orders_by_status.v1"], start: date, end: date
) -> list[dict]:
    """Count orders by status from start (inclusive) to end (exclusive), in UTC."""
    if query_id != SALES_STATUS_QUERY or not list_saved_queries():
        raise ValueError("Query not permitted")
    if not 0 < (end - start).days <= 366:
        raise ValueError("Expected an increasing date range of at most 366 days")
    return aggregate_records("sale.order", "state", domain=[
        ["date_order", ">=", f"{start.isoformat()} 00:00:00"],
        ["date_order", "<", f"{end.isoformat()} 00:00:00"],
    ])


class RequireProfile:
    def __init__(self, downstream, profiles):
        self.downstream = downstream
        self.profiles = profiles

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.downstream(scope, receive, send)
        headers = [v for k, v in scope["headers"] if k.lower() == b"authorization"]
        scheme, _, token = headers[0].partition(b" ") if len(headers) == 1 else (b"", b"", b"")
        profile = next(
            (p for p in self.profiles if scheme.lower() == b"bearer" and compare_digest(token, p.token)),
            None,
        )
        if profile is None:
            response = JSONResponse({"error": "Unauthorized"}, 401, headers={"WWW-Authenticate": "Bearer"})
            return await response(scope, receive, send)
        marker = principal.set(profile)
        try:
            await self.downstream(scope, receive, send)
        finally:
            principal.reset(marker)


def create_app():
    profiles = configured_profiles()
    hosts = os.getenv("ODOO_MCP_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if not all(re.fullmatch(r"[a-zA-Z0-9.-]+(?::[0-9]{1,5})?", host) for host in hosts):
        raise ValueError("Expected exact Host headers, separated by commas")
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=[]
    )
    return RequireProfile(mcp.streamable_http_app(), profiles)
