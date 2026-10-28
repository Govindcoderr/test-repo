"""Resolve a tool name to a runnable catalog entry.

Pure lookup: no execution, no I/O. A miss is a normal outcome rather than an
error, so callers turn it into a SKIPPED result via ``core/errors.py``.

This is also the seam where unsupported tool sources are rejected. The catalog can
describe an MCP tool before the MCP transport exists, and resolving it should fail
cleanly rather than reaching a half-built code path.
"""

from ai_os.app.infrastructure.engines.tool_engine.catalog.loader import ToolCatalog
from ai_os.app.infrastructure.engines.tool_engine.domain.catalog import CatalogEntry
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import ToolSource

SUPPORTED_SOURCES: frozenset[ToolSource] = frozenset({ToolSource.NATIVE, ToolSource.MCP, ToolSource.API})
"""Sources with a working execution path. Any source added to the catalog without
being added here fails cleanly rather than reaching a half-built code path."""


def resolve(catalog: ToolCatalog, tool_name: str) -> CatalogEntry | None:
    """Return the entry for ``tool_name``, or None if unknown or disabled."""
    return catalog.get(tool_name)


def is_supported(entry: CatalogEntry) -> bool:
    """True if this engine has a working execution path for the entry's source."""
    return entry.source in SUPPORTED_SOURCES
