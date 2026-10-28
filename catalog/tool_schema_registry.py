"""Unified tool schema registry loader.

This is the single source of truth for all tool schemas — NATIVE, MCP, and API.
Both the tool-engine-service and the agent-runtime-service read from this module.

Schema lifetime
---------------
Schemas are loaded once at service startup and held in memory for the lifetime
of the process. No file I/O happens at request time — the Worker receives schemas
via the manifest metadata (toolSchemas), which is built once and cached in the
kernel's EngineRegistry until the next service restart.

If a schema changes:
  1. Update tool_schemas.json
  2. Restart tool-engine-service (rebuilds manifest with new schemas)
  3. Restart kernel-service (re-fetches manifest from tool-engine at startup)

No agent-runtime restart needed — schemas flow to it via the kernel injection
on every agent.* dispatch, not from its own cache.

Database migration path
-----------------------
When you're ready to move schemas to a database, replace the body of
``load_tool_schemas()`` with a DB fetch. The function signature stays identical.
Every caller (tool-engine catalog loader, agent-runtime ToolCatalog) keeps
working without any other changes — this is the only function to swap.

    # Future DB version:
    async def load_tool_schemas(path=None):
        async with db.connect() as conn:
            rows = await conn.fetch("SELECT * FROM tool_schemas WHERE enabled=true")
            return [dict(row) for row in rows]

The synchronous signature stays for now since both callers run at startup
before the async event loop is needed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("tool-engine.catalog.tool_schema_registry")

# Default path — sits next to this file in the catalog/ directory.
# Override with TOOL_SCHEMAS_PATH env var for deployments where the
# agent-runtime-service is on a different host than the tool-engine-service
# (e.g. point to a shared volume, or a copy bundled into the agent-runtime image).
DEFAULT_SCHEMAS_PATH = Path(__file__).parent / "tool_schemas.json"
ENV_SCHEMAS_PATH = "TOOL_SCHEMAS_PATH"


class ToolSchemaRegistryError(RuntimeError):
    """Raised when tool_schemas.json cannot be loaded or is malformed."""


def resolved_schemas_path() -> Path:
    """Return the schemas file path, honouring ``TOOL_SCHEMAS_PATH``."""
    override = os.environ.get(ENV_SCHEMAS_PATH)
    return Path(override).expanduser() if override else DEFAULT_SCHEMAS_PATH


def _normalize_api_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Apply the API tool schema shape to one entry.

    API tool shape (one entry per operation)::

        {
          "toolId": "01",                 # id of the parent tool
          "name": "GMAIL_get_many",       # callable key: <TOOL>_<operation>
          "toolName": "GMAIL",            # parent tool this operation belongs to
          "category": "",
          "source": "API",
          "description": "...",
          "required": [...],
          "properties": {...},
          "dagConfig": {value, name, expressionExecutionName,
                        defaultParameters, requiresAuth, credentialId}
        }

    ``name`` is what the Worker calls and what the Tool Engine catalog is
    keyed on. ``toolName`` groups operations under one tool (it maps to the
    ``allowedTools[*].name`` group whose ``operation`` list holds the
    ``name`` values). Missing identity fields are defaulted so older
    entries keep loading; a missing ``dagConfig`` is logged because such a
    tool cannot be executed.
    """
    name = entry["name"]
    normalized = {
        **entry,
        "toolId": str(entry["toolId"]) if entry.get("toolId") is not None else "",
        "toolName": entry.get("toolName") or name,
        "category": entry.get("category") or "",
    }
    if not isinstance(entry.get("dagConfig"), dict) or not entry["dagConfig"]:
        logger.warning("API tool %r has no dagConfig; it cannot be executed", name)
    return normalized


def load_tool_schemas(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all tool schema entries from tool_schemas.json.

    Returns a flat list of tool entry dicts. Each dict has at minimum:
      - ``name``   : tool name (e.g. "web_fetch", "GMAIL_send")
      - ``source`` : "NATIVE" | "MCP" | "API"

    NATIVE/MCP entries additionally have ``required`` and ``properties``.
    API entries additionally have ``toolId``, ``toolName``, ``category`` and
    ``dagConfig`` with backend DAG node config (see ``_normalize_api_entry``).

    Raises ``ToolSchemaRegistryError`` if the file is missing or malformed.
    A missing file is treated as a fatal startup error — the registry is
    authoritative and must exist.
    """
    target = path or resolved_schemas_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolSchemaRegistryError(
            f"Cannot read tool schemas at {target}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolSchemaRegistryError(
            f"tool_schemas.json at {target} is not valid JSON: {exc}"
        ) from exc

    tools = data.get("tools")
    if not isinstance(tools, list):
        raise ToolSchemaRegistryError(
            f"tool_schemas.json at {target} has no 'tools' array"
        )

    validated: list[dict[str, Any]] = []
    for i, entry in enumerate(tools):
        if not isinstance(entry, dict):
            raise ToolSchemaRegistryError(
                f"tool_schemas.json tools[{i}] is not an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ToolSchemaRegistryError(
                f"tool_schemas.json tools[{i}] is missing a 'name'"
            )
        source = entry.get("source", "NATIVE").upper()
        if source not in ("NATIVE", "MCP", "API"):
            raise ToolSchemaRegistryError(
                f"tool_schemas.json tools[{i}] ({name}) has unknown source {source!r}"
            )
        normalized: dict[str, Any] = {**entry, "source": source}
        if source == "API":
            normalized = _normalize_api_entry(normalized)
        validated.append(normalized)

    seen_names: set[str] = set()
    for entry in validated:
        if entry["name"] in seen_names:
            raise ToolSchemaRegistryError(
                f"tool_schemas.json has duplicate tool name {entry['name']!r}"
            )
        seen_names.add(entry["name"])

    logger.info(
        "Loaded %d tool schema entries from %s (%d NATIVE, %d MCP, %d API)",
        len(validated),
        target,
        sum(1 for e in validated if e["source"] == "NATIVE"),
        sum(1 for e in validated if e["source"] == "MCP"),
        sum(1 for e in validated if e["source"] == "API"),
    )
    return validated


def get_native_mcp_schemas(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{tool_name: input_schema}`` for NATIVE and MCP tools only.

    Used by the tool-engine catalog loader to inject schemas into
    ``CatalogEntry.input_schema`` for tools that have entries in
    tool_schemas.json. MCP tools discovered dynamically at runtime
    will have their schemas overridden by what the MCP server reports —
    entries here act as a baseline/correction (same role as schema_overrides).
    """
    entries = load_tool_schemas(path)
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["source"] not in ("NATIVE", "MCP"):
            continue
        schema: dict[str, Any] = {}
        if entry.get("required"):
            schema["required"] = list(entry["required"])
        if entry.get("properties"):
            schema["properties"] = dict(entry["properties"])
        if schema:
            schema.setdefault("type", "object")
            result[entry["name"]] = schema
    return result


def get_api_tool_configs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{tool_name: dag_config_dict}`` for API tools only.

    Used by the agent-runtime ToolCatalog to load REST tool descriptors
    without hardcoding them in Python. Each value is the ``dagConfig``
    block from the schema entry, with ``name``, ``description``,
    ``required``, and ``properties`` merged in for completeness.
    """
    entries = load_tool_schemas(path)
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["source"] != "API":
            continue
        dag_config = entry.get("dagConfig") or {}
        result[entry["name"]] = {
            # Tool identity
            "tool_id": entry.get("toolId", ""),
            "tool_name": entry.get("toolName") or entry["name"],
            "category": entry.get("category", ""),
            # DAG node config (required for dag_builder)
            "value": dag_config.get("value", entry["name"]),
            "name": dag_config.get("name", entry.get("description", entry["name"])),
            "expression_execution_name": dag_config.get(
                "expressionExecutionName",
                dag_config.get("name", entry["name"]),
            ),
            "default_parameters": dict(dag_config.get("defaultParameters") or {}),
            "requires_auth": bool(dag_config.get("requiresAuth", False)),
            "credential_id": dag_config.get("credentialId"),
            # Argument schema (for Worker prompt)
            "description": entry.get("description", ""),
            "required": list(entry.get("required") or []),
            "properties": dict(entry.get("properties") or {}),
        }
    return result


def get_all_schemas_for_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return compact ``{tool_name: {required, properties, source}}`` for ALL tools.

    Used by the tool-engine manifest builder to put schemas into
    ``manifest.metadata["toolSchemas"]``. The kernel reads this and
    enriches it further with source from allowedTools before injecting
    into agent.* payloads.

    API tools are included here so the Worker can see their argument
    schemas even though they are not in the native/MCP catalog.
    """
    entries = load_tool_schemas(path)
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        props = entry.get("properties") or {}
        compact_props = {
            key: {
                "type": spec.get("type", "string"),
                "description": str(spec.get("description") or "")[:120],
            }
            for key, spec in props.items()
            if isinstance(spec, dict)
        }
        schema: dict[str, Any] = {
            "source": entry["source"],
        }
        if entry.get("required"):
            schema["required"] = list(entry["required"])
        if compact_props:
            schema["properties"] = compact_props
        if entry.get("description"):
            schema["description"] = entry["description"]
        result[entry["name"]] = schema
    return result
