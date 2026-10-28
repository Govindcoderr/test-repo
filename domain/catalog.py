"""The shape of one entry in the tool catalog.

This is the engine's *internal* registry, distinct from the kernel's capability
registry:

- Kernel capability registry advertises ``tool.execute`` plus this engine's
  ``resources`` so the kernel can route to it. Built in ``tool_engine/manifest.py``.
- Tool catalog (this) maps a tool name to something runnable inside the engine.
  Loaded by ``tool_engine/catalog/loader.py``.

The two are linked: the manifest's ``resources`` list is derived from the catalog,
so a tool is declared in exactly one place.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_os.app.infrastructure.engines.tool_engine.domain.enums import ToolSource


class CatalogEntry(BaseModel):
    """One tool the engine knows how to run."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    """Name callers use. For MCP tools this is namespaced as
    ``{server}.{remoteTool}`` so two servers cannot collide."""

    source: ToolSource
    target: str
    """Implementation reference. For NATIVE this is the registered function name,
    for MCP the remote tool name as the server advertises it."""

    server: str | None = None
    """MCP server this tool came from. None for NATIVE tools."""

    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")
    required_permission: str | None = Field(default=None, alias="requiredPermission")
    """Permission the caller must hold. None means the tool is unrestricted.
    Enforced by ``core/permissions.py`` because the kernel only authorizes at
    capability level."""

    # API-source tool identity. ``name`` is the per-operation callable key
    # (e.g. "GMAIL_send"); ``tool_name`` is the parent tool it belongs to
    # (e.g. "GMAIL") and ``tool_id`` the parent's id. All optional so
    # NATIVE/MCP entries are unaffected.
    tool_id: str | None = Field(default=None, alias="toolId")
    tool_name: str | None = Field(default=None, alias="toolName")
    category: str = ""

    # Present on API-source tools only. Carries the backend DAG node config:
    # value, name, expressionExecutionName, defaultParameters, requiresAuth,
    # credentialId. Read by executor._build_api_dag to construct the REST payload.
    dag_config: dict[str, Any] | None = Field(default=None, alias="dagConfig")

    enabled: bool = True

    discovered: bool = False
    """True if this entry came from asking a server what it offers, rather than from
    ``tools.json``.

    Re-discovery replaces discovered entries and leaves the rest alone. Keying that on
    a flag rather than on ``source`` matters: keying on source would silently delete a
    hand-written MCP entry the moment any server was discovered."""


class ToolCatalogFile(BaseModel):
    """Top-level shape of ``catalog/tools.json``."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    version: int = 1
    tools: list[CatalogEntry] = Field(default_factory=list)
