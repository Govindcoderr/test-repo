"""Engine settings.

PLACEHOLDER - not yet implemented.

``pydantic-settings`` is already installed and is the natural fit here.

Keep this small. Much of what looks like engine configuration actually arrives per
request from the kernel - timeouts (``runtimeConfiguration.maxExecutionTime``),
retry behaviour (``autoRetry``), debug flags (``debugMode``) - so duplicating them
as local settings would create two sources of truth. Settings here should be
deployment-level only: catalog path, MCP server config location, log level.
"""
