"""Per-tool permission checks.

The kernel's ``AuthorizationService.authorize()`` decides at capability level
(``tool.execute``) and its default implementation allows everything. It cannot
express "this agent may call crm.lookup but not customer.delete", so per-tool
enforcement happens here.

Kept as a separate module rather than inlined into the executor so that if the
platform later moves this decision to the kernel or a Tool Gateway, the layer can
be removed without touching execution logic.
"""

from ai_os.app.infrastructure.engines.tool_engine.domain.catalog import CatalogEntry


def is_allowed(entry: CatalogEntry, granted: list[str]) -> bool:
    """True if the caller may run this tool.

    An entry with no ``required_permission`` is unrestricted.
    """
    if entry.required_permission is None:
        return True
    return entry.required_permission in granted
