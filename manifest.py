"""Engine manifest data.

Declared with the engine's own model rather than the kernel's ``EngineManifest``
so this module stays importable without the kernel repo on the path. The adapter in
``kernel_adapter.py`` converts it.

Field names match the kernel's ``EngineManifest`` one-for-one, so the conversion is
a direct mapping with no renaming.

``resources`` is derived from the catalog rather than hardcoded, so adding a tool
never means editing this file. Once registered, the tool becomes routable through
``GET /api/v1/kernel/registry`` with no kernel-side change.

``metadata`` carries ``toolSchemas`` — the input schemas for every enabled tool
that publishes one. The Agent Runtime's Worker reads this via the kernel's engine
registry (injected on every agent.* dispatch as ``toolSchemas``) so it knows the
exact argument names and types for each tool, rather than guessing them from the
tool name alone.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_os.app.infrastructure.engines.tool_engine.catalog.loader import ToolCatalog

ENGINE_ID = "tool-engine"
ENGINE_NAME = "Tool Engine"
ENGINE_VERSION = "0.1.0"
CAPABILITY_TOOL_EXECUTE = "tool.execute"
TOOL_SCHEMAS_KEY = "toolSchemas"


class EngineManifestData(BaseModel):
    """Local mirror of the kernel's EngineManifest fields."""

    model_config = ConfigDict(populate_by_name=True)

    engine_id: str = Field(alias="engineId")
    name: str
    version: str
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    priority: int = 100


def build_manifest(catalog: ToolCatalog, priority: int = 100) -> EngineManifestData:
    """Build the manifest, advertising every enabled tool as a resource.

    ``metadata[toolSchemas]`` carries the input schemas for tools that publish one.
    The kernel injects these directly onto agent.* dispatches so the Agent Runtime
    Worker knows the exact argument names/types without a separate HTTP round trip.
    """
    return EngineManifestData(
        engine_id=ENGINE_ID,
        name=ENGINE_NAME,
        version=ENGINE_VERSION,
        description="Executes native and MCP tools on behalf of agents.",
        capabilities=[CAPABILITY_TOOL_EXECUTE],
        resources=catalog.names(),
        metadata={TOOL_SCHEMAS_KEY: catalog.schemas()},
        priority=priority,
    )
