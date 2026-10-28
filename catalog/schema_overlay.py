"""Read the hand-authored schema overlay and hand it to the catalog.

Why a merge and not a replacement
---------------------------------
MCP tool schemas are discovered, not written down: ``engine.discover_mcp_tools()``
asks each server what it offers and stores each ``inputSchema`` on the matching
``CatalogEntry``. That is the right source of truth for *shape* — it cannot drift
from the server the way a copy in this repo would.

What discovery cannot fix is that the Worker is an LLM. It sees the schema as prose
in a prompt, and a schema that is accurate but terse still leads it to send ``range``
where the server wants ``range_name``, because nothing in the discovered text says
which spelling is the real one. So this layer corrects descriptions and required
lists for the few fields that provably mislead, and leaves everything else discovery
reported intact. ``ToolCatalog.apply_schema_overlay`` does the merging; this module
only parses.

Where the file lives
--------------------
``catalog/schema_overrides.json``, beside the catalog it belongs with — it is
operator-editable data, not code. Override the location with the
``TOOL_ENGINE_SCHEMA_OVERLAY_FILE`` environment variable.

Failure behaviour
-----------------
A missing overlay file is fine — it means no corrections, and discovery's schemas
stand. A file that exists but is malformed raises, matching ``loader.py``'s rule:
broken configuration is a deployment fault and must not degrade into a per-request
tool failure that nobody can trace back to a JSON typo.

Overlay file format
-------------------
``{"schemas": [{"name": "tool_name", "parameters": {<json_schema>}}, ...]}``
"""

import json
import os
from pathlib import Path
from typing import Any

_CATALOG_DIR = Path(__file__).resolve().parent

DEFAULT_OVERLAY_PATH = _CATALOG_DIR / "schema_overrides.json"
ENV_OVERLAY_FILE = "TOOL_ENGINE_SCHEMA_OVERLAY_FILE"


class SchemaOverlayError(RuntimeError):
    """The overlay file exists but could not be read or is invalid."""


def resolved_overlay_file() -> Path:
    """The overlay path, honouring ``TOOL_ENGINE_SCHEMA_OVERLAY_FILE``."""
    override = os.environ.get(ENV_OVERLAY_FILE)
    return Path(override).expanduser() if override else DEFAULT_OVERLAY_PATH


def load_overlay(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{tool_name: json_schema}`` from the overlay file.

    An absent file returns an empty mapping — no corrections, catalog schemas stand.
    Each value is the entry's ``parameters`` block verbatim (already JSON Schema),
    so callers can pass it straight to ``ToolCatalog.apply_schema_overlay()``.
    """
    target = path or resolved_overlay_file()
    if not target.exists():
        return {}

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaOverlayError(f"cannot read schema overlay at {target}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaOverlayError(f"schema overlay at {target} is not valid JSON: {exc}") from exc

    entries = data.get("schemas")
    if not isinstance(entries, list):
        raise SchemaOverlayError(f"schema overlay at {target} has no 'schemas' array")

    overlay: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SchemaOverlayError(f"{target} schemas[{index}] is not an object")

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise SchemaOverlayError(f"{target} schemas[{index}] is missing a 'name'")
        if name in overlay:
            raise SchemaOverlayError(f"{target} defines {name!r} more than once")

        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            raise SchemaOverlayError(
                f"{target} schemas[{index}] ({name}) has no 'parameters'"
            )

        overlay[name] = parameters

    return overlay
