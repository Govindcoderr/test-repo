"""Load and index the tool catalog, preserving full published input schemas.

The manifest advertises definitions; the Kernel filters them per execution.
Schema size is reduced by selecting tools, not by discarding their constraints.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai_os.app.infrastructure.engines.tool_engine.catalog.tool_schema_registry import (
    get_native_mcp_schemas,
    get_all_schemas_for_manifest,
    ToolSchemaRegistryError,
)
from ai_os.app.infrastructure.engines.tool_engine.domain.catalog import CatalogEntry, ToolCatalogFile

DEFAULT_CATALOG_PATH = Path(__file__).parent / "tools.json"


class CatalogError(RuntimeError):
    """Raised when the catalog cannot be loaded or is invalid."""


def _merge_schema(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge schema objects; overlay lists/scalars replace originals.

    A property-description correction preserves that property's nested schema.
    Explicit required lists, combinator arrays and false constraints replace the
    original values. Neither source object is mutated or shared with the result.
    """
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_schema(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _input_overlay(schema: dict[str, Any]) -> dict[str, Any]:
    """Accept a raw input schema or a definition with inputSchema.

    A wrapper's routing/configuration fields are never merged into input schema.
    """
    if "inputSchema" in schema:
        value = schema["inputSchema"]
        if not isinstance(value, dict):
            raise CatalogError("Overlay inputSchema must be an object")
        return value
    return schema


class ToolCatalog:
    """In-memory, name-indexed view of the tool catalog."""

    def __init__(self, entries: list[CatalogEntry]) -> None:
        self._by_name: dict[str, CatalogEntry] = {}
        for entry in entries:
            if entry.name in self._by_name:
                raise CatalogError(f"duplicate tool name in catalog: {entry.name!r}")
            self._by_name[entry.name] = entry

    @classmethod
    def from_file(cls, path: Path | None = None) -> "ToolCatalog":
        catalog_path = path or DEFAULT_CATALOG_PATH
        try:
            raw = catalog_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CatalogError(f"cannot read catalog at {catalog_path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"catalog at {catalog_path} is not valid JSON: {exc}") from exc
        try:
            parsed = ToolCatalogFile.model_validate(data)
        except ValidationError as exc:
            raise CatalogError(f"catalog at {catalog_path} is invalid: {exc}") from exc

        catalog = cls(parsed.tools)
        # Preserve the existing optional-registry behavior. A registry load
        # failure is not repaired here; the Kernel rejects missing selections.
        try:
            registry_schemas = get_native_mcp_schemas()
            if registry_schemas:
                catalog.apply_schema_overlay(registry_schemas)
        except ToolSchemaRegistryError:
            pass
        return catalog

    def register_dynamic(self, entries: list[CatalogEntry]) -> int:
        """Replace discovered tools, retaining handwritten catalog entries.

        Validate collisions before committing so failed discovery cannot leave
        a partially updated catalog.
        """
        updated = {
            name: entry for name, entry in self._by_name.items() if not entry.discovered
        }
        for entry in entries:
            if entry.name in updated:
                raise CatalogError(
                    f"discovered tool {entry.name!r} collides with an existing entry"
                )
            updated[entry.name] = entry.model_copy(update={"discovered": True})
        self._by_name = updated
        return sum(1 for e in self._by_name.values() if e.discovered)

    def apply_schema_overlay(self, overlay: dict[str, dict]) -> int:
        """Apply explicit corrections without discarding unrelated schema data.

        Unknown tools are ignored, as before. Objects merge recursively; lists
        and scalars replace existing values. In particular, required replaces
        rather than appends, and root constraints such as oneOf, if/then and
        additionalProperties are retained. Wrappers with inputSchema are also
        accepted. This does not register tools or change their execution source.
        """
        updated = dict(self._by_name)
        applied = 0
        for name, schema in overlay.items():
            entry = updated.get(name)
            if entry is None:
                continue
            if not isinstance(schema, dict):
                raise CatalogError(f"Schema overlay for {name!r} must be an object")
            base = entry.input_schema or {}
            if not isinstance(base, dict):
                raise CatalogError(f"Input schema for {name!r} must be an object")
            patch = deepcopy(_input_overlay(schema))
            # Retain the old treatment of required:null as an empty list.
            if "required" in patch and patch["required"] is None:
                patch["required"] = []
            merged = _merge_schema(base, patch)
            merged.setdefault("type", "object")
            updated[name] = entry.model_copy(update={"input_schema": merged})
            applied += 1
        self._by_name = updated
        return applied

    def get(self, name: str) -> CatalogEntry | None:
        """Return an enabled entry, or None for an unknown/disabled tool."""
        entry = self._by_name.get(name)
        if entry is None or not entry.enabled:
            return None
        return entry

    def names(self) -> list[str]:
        """Enabled catalog tool names; API-only names still live in the registry."""
        return sorted(name for name, e in self._by_name.items() if e.enabled)

    def schemas(self) -> dict[str, dict]:
        """Full definitions for the manifest, copied without truncation.

        Keep the existing top-level required/properties layout for catalog tools,
        plus EVERY other input-schema keyword (nested properties, items, $defs,
        $ref, anyOf/oneOf, enums, bounds, defaults and conditional constraints).
        Include the catalog description and execution source when available.

        API definitions are copied as supplied by the registry. This method
        cannot restore information already removed by that upstream helper.
        Catalog entries take precedence, including disabled entries: a disabled
        catalog name must not reappear through an API registry collision.
        """
        result: dict[str, dict] = {}
        try:
            all_registry = get_all_schemas_for_manifest()
            for name, schema in all_registry.items():
                source = schema.get("source")
                source = getattr(source, "value", source)
                if isinstance(source, str) and source.upper() == "API":
                    result[name] = deepcopy(schema)
        except ToolSchemaRegistryError:
            pass

        for name, entry in self._by_name.items():
            result.pop(name, None)
            if not entry.enabled or not entry.input_schema:
                continue
            if not isinstance(entry.input_schema, dict):
                raise CatalogError(f"Input schema for {name!r} must be an object")
            definition = deepcopy(entry.input_schema)
            description = getattr(entry, "description", None)
            if description:
                definition["description"] = str(description)
            source = getattr(entry, "source", None)
            source = getattr(source, "value", source)
            if source is not None:
                definition["source"] = str(source).upper()
            result[name] = definition
        return result

    def __len__(self) -> int:
        return len(self._by_name)
