"""Execute a single tool call and report what happened.

This is the exception boundary. Catching broad ``Exception`` is correct *here
specifically*, because the engine must never raise at the kernel edge. Every
outcome, including failure and timeout, comes back as a ``ToolCallResult`` with a
measured duration.

``asyncio.CancelledError`` is deliberately not caught. It inherits from
``BaseException``, not ``Exception``, so it propagates - which is what an ABORT in
a parallel batch relies on to stop work that is still in flight.

No retry logic lives here. ``KernelRequest.retry_policy`` and the kernel's
``retry_manager`` own retries; this layer only executes once and reports.

Credential injection
--------------------
For MCP tools the executor resolves credentials with this priority:

  1. ``call.tool_config`` — credentials from the incoming payload
     (``toolConfig.tokens.access_token``).  This is the primary production
     path: the kernel injects credentials from the backend store into the
     payload so the Tool Engine is stateless.

  2. ``CredentialManager`` — Redis-based lookup by ``user_id``.  Used as a
     fallback when ``toolConfig`` is absent (e.g. during local dev/test
     where the kernel has not yet been updated to inject credentials).

The MCP client binds ``user_google_email`` to the resolved account only
when the authenticated tool's live schema declares that argument. OAuth2
authentication alone does not determine a tool's input parameters.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ai_os.app.infrastructure.engines.tool_engine.catalog.loader import ToolCatalog
from ai_os.app.infrastructure.engines.tool_engine.core import errors, permissions, references
from ai_os.app.infrastructure.engines.tool_engine.core.circuit_breaker import CircuitBreaker
from ai_os.app.infrastructure.engines.tool_engine.core.constants import DEFAULT_TIMEOUT_SECONDS
from ai_os.app.infrastructure.engines.tool_engine.core.resolver import is_supported, resolve
from ai_os.app.infrastructure.engines.tool_engine.domain.catalog import CatalogEntry
from ai_os.app.infrastructure.engines.tool_engine.domain.contracts import (
    ToolCall,
    ToolCallResult,
    ToolError,
)
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import ToolSource, ToolStatus
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.credentials.models import (
    CredentialNotFoundError,
    InvalidCredentialConfigError,
    UserCredential,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway import (
    client as mcp_client,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.config import (
    MCPConfigError,
    MCPServerConfig,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.native import runner
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.rest_execution import (
    client as rest_client,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.rest_execution.config import (
    RestExecutionSettings,
    get_settings as get_rest_settings,
)

if TYPE_CHECKING:
    from ai_os.app.infrastructure.engines.tool_engine.infrastructure.credentials.credential_manager import (
        CredentialManager,
    )

logger = logging.getLogger(__name__)

_DRIVE_URL_ID_RE = re.compile(r"(?:/d/|/folders/|[?&]id=)([a-zA-Z0-9_-]{10,})")


def _extract_drive_id(value: str) -> str:
    """Reduce a full Drive/Docs URL to its bare file/folder ID.

    The Worker is instructed (WORKER_SYSTEM_PROMPT, Agent-Runtime-Service)
    to do this extraction itself before calling a Drive tool, but LLM
    compliance with that instruction isn't guaranteed — this is the
    deterministic backstop so a stray full URL doesn't fail the call or
    trigger a HITL. A value that isn't a recognizable Drive/Docs URL
    (e.g. already a bare ID) is returned unchanged.
    """
    match = _DRIVE_URL_ID_RE.search(value)
    return match.group(1) if match else value


def breaker_key(tool_name: str) -> str:
    """The circuit breaker key for a tool — one key per tool name."""
    return tool_name


async def execute_call(
    call: ToolCall,
    catalog: ToolCatalog,
    granted_permissions: list[str],
    timeout_seconds: int,
    outputs: dict[str, dict | None] | None = None,
    mcp_configs: dict[str, MCPServerConfig] | None = None,
    breaker: CircuitBreaker | None = None,
    user_id: str | None = None,
    credential_manager: "CredentialManager | None" = None,
    rest_settings: RestExecutionSettings | None = None,
    disabled_servers: frozenset[str] = frozenset(),
) -> ToolCallResult:
    """Run one tool call. Never raises except on cancellation.

    ``outputs`` maps already-completed ``toolCallId`` values to their output,
    supplied only in SEQUENTIAL mode. ``$ref`` placeholders in the input are
    substituted before the tool runs.

    ``mcp_configs`` holds the server configs from discovery, keyed by name.

    Credential resolution (MCP tools only):
      1. ``call.tool_config`` → ``UserCredential.from_tool_config()``
         (payload-injected credentials — primary production path)
      2. ``credential_manager.get_credential(user_id, server_config)``
         (Redis fallback — used when toolConfig is absent)
    """
    started = perf_counter()

    def elapsed_ms() -> float:
        return round((perf_counter() - started) * 1000, 3)

    # ── $ref resolution ──────────────────────────────────────────────────────
    if outputs is not None and references.contains_reference(call.input):
        try:
            resolved_input = references.resolve(call.input, outputs)
        except Exception as exc:
            status, error = errors.classify(exc)
            return _result(call, status, error=error, duration_ms=elapsed_ms())
        call = call.model_copy(update={"input": resolved_input})

    # ── Provider kill switch ─────────────────────────────────────────────────
    prefix = call.tool_name.split(".", 1)[0] if "." in call.tool_name else None
    if prefix and prefix in disabled_servers:
        status, error = errors.provider_disabled(prefix)
        return _result(call, status, error=error, duration_ms=elapsed_ms())

    # ── Catalog lookup ───────────────────────────────────────────────────────
    entry = resolve(catalog, call.tool_name)
    if entry is None:
        status, error = errors.tool_not_found(call.tool_name)
        return _result(call, status, error=error, duration_ms=elapsed_ms())

    if not is_supported(entry):
        return _result(
            call,
            ToolStatus.SKIPPED,
            error=ToolError(
                code="SOURCE_NOT_SUPPORTED",
                message=f"tool source {entry.source.value} has no execution path yet",
            ),
            duration_ms=elapsed_ms(),
        )

    if call.tool_name.startswith("google_workspace."):
        normalized_input = dict(call.input)
        changed = False
        for key in ("file_id", "folder_id"):
            value = normalized_input.get(key)
            if isinstance(value, str):
                normalized = _extract_drive_id(value)
                if normalized != value:
                    normalized_input[key] = normalized
                    changed = True
        if changed:
            call = call.model_copy(update={"input": normalized_input})
            logger.debug("Normalized Drive URL to bare ID for tool=%s", call.tool_name)

    # ── Permission check ─────────────────────────────────────────────────────
    if not permissions.is_allowed(entry, granted_permissions):
        assert entry.required_permission is not None
        status, error = errors.permission_denied(call.tool_name, entry.required_permission)
        return _result(call, status, error=error, duration_ms=elapsed_ms())

    # ── Credential resolution (MCP tools only) ───────────────────────────────
    credential: UserCredential | None = None
    effective_source = _effective_source(entry, call)

    if effective_source is ToolSource.MCP and mcp_configs:
        server_name = entry.server
        server_config = mcp_configs.get(server_name) if server_name else None

        if server_config and server_config.auth_type != "none":
            try:
                credential, missing_reason = await _resolve_credential(
                    call, server_config, user_id, credential_manager,
                )
            except InvalidCredentialConfigError as exc:
                status, error = errors.invalid_credential(exc.code, server_config.name, str(exc))
                return _result(call, status, error=error, duration_ms=elapsed_ms())

            rejection = _precheck(credential, server_config, missing_reason)
            if rejection is not None:
                status, error = rejection
                logger.info(
                    "Credential pre-check rejected tool=%s server=%s code=%s",
                    call.tool_name, server_config.name, error.code,
                )
                return _result(call, status, error=error, duration_ms=elapsed_ms())

            # ── user_google_email inject ──────────────────────────────────
            # EXTERNAL_OAUTH21_PROVIDER mode: workspace-mcp extracts the email
            # from the Bearer token itself — do NOT inject user_google_email,
            # it causes a pydantic validation error ("unexpected keyword argument").
            # The Bearer token already carries the user identity.
            # Only strip it from input if LLM provided it — server will reject it.
            if (
                credential is not None
                and credential.auth_type == "oauth2"
                and "user_google_email" in call.input
            ):
                new_input = {k: v for k, v in call.input.items() if k != "user_google_email"}
                call = call.model_copy(update={"input": new_input})
                logger.debug("Removed user_google_email from input (EXTERNAL_OAUTH21_PROVIDER mode)")

            # ── Priority 1: toolConfig from payload ───────────────────────
            # Agent Runtime injects allowedTools[*].configuration as toolConfig.
            # This is the stateless production path — no Redis, no file I/O.
            if call.tool_config:
                try:
                    credential = UserCredential.from_tool_config(call.tool_config)
                except (TypeError, ValueError) as exc:
                    return _result(
                        call,
                        ToolStatus.DENIED,
                        error=ToolError(
                            code="NOT_AUTHENTICATED",
                            message=str(exc),
                            retryable=False,
                        ),
                        duration_ms=elapsed_ms(),
                    )
                if credential:
                    logger.debug(
                        "Credential from payload toolConfig for tool=%s email=%s",
                        call.tool_name, credential.email,
                    )

            # ── Priority 2: CredentialManager (Redis fallback) ───────────
            # Used during local dev/test or when kernel hasn't injected toolConfig.
            if credential is None and credential_manager:
                try:
                    credential = await credential_manager.get_credential(
                        user_id, server_config
                    )
                    logger.debug(
                        "Credential from CredentialManager for user=%s tool=%s",
                        user_id, call.tool_name,
                    )
                except CredentialNotFoundError as exc:
                    logger.warning(
                        "Credential not found: user=%s server=%s — %s",
                        user_id, server_name, exc,
                    )
                    return _result(
                        call,
                        ToolStatus.DENIED,
                        error=ToolError(
                            code="NOT_AUTHENTICATED",
                            message=str(exc),
                            retryable=False,
                        ),
                        duration_ms=elapsed_ms(),
                    )

            # The MCP client normalizes account identity using the live tool
            # schema after connecting with this credential. Do not inject or
            # strip tool parameters based solely on the OAuth authentication type.

    # ── Circuit breaker gate ─────────────────────────────────────────────────
    key = breaker_key(call.tool_name)
    if breaker is not None and not await breaker.allow(key):
        status, error = errors.circuit_open(call.tool_name, key)
        return _result(call, status, error=error, duration_ms=elapsed_ms())

    # ── Tool dispatch ────────────────────────────────────────────────────────
    try:
        output = await asyncio.wait_for(
            _dispatch(
                entry, call, mcp_configs or {}, credential, catalog, user_id,
                credential_manager, timeout_seconds, granted_permissions,
                rest_settings or get_rest_settings(),
                disabled_servers=disabled_servers,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        if breaker is not None:
            await breaker.record_failure(key, retryable=True)
        return _result(
            call,
            ToolStatus.TIMEOUT,
            error=ToolError(
                code="TOOL_TIMEOUT",
                message="tool exceeded the allotted timeout",
                retryable=True,
            ),
            duration_ms=elapsed_ms(),
        )
    except Exception as exc:
        status, error = errors.classify(exc)
        if breaker is not None:
            # Only outages count; auth failures and 429s must not pause the
            # tool for every other user.
            await breaker.record_failure(key, retryable=errors.counts_toward_breaker(error))
        return _result(call, status, error=error, duration_ms=elapsed_ms())

    # Success
    if breaker is not None:
        await breaker.record_success(key)

    return _result(call, ToolStatus.SUCCESS, output=output, duration_ms=elapsed_ms())


_EXPIRY_BUFFER_SECONDS = 30.0


async def _resolve_credential(
    call: ToolCall,
    server_config: MCPServerConfig,
    user_id: str | None,
    credential_manager: "CredentialManager | None",
) -> tuple[UserCredential | None, str | None]:
    """Find the credential for an MCP call.

    Order: (a) ``call.toolConfig`` — the stateless production path where
    Agent Runtime injects the token from ``allowedTools[*].configuration``.
    (b) The CredentialManager fallback — dev only, used when toolConfig is
    absent (``CREDENTIAL_FALLBACK_ENABLED=true`` in dev .env).

    Returns ``(credential, reason)``; ``reason`` explains a missing credential.
    Raises ``InvalidCredentialConfigError`` for a malformed payload config.
    """
    # (a) per-call toolConfig: the stateless production path.
    credential = UserCredential.from_tool_config(call.tool_config)
    if credential is not None:
        return credential, None

    # (b) dev-only fallback (file/Redis TokenStore).
    if credential_manager is None:
        return None, None
    try:
        credential = await credential_manager.get_credential(user_id, server_config)
    except CredentialNotFoundError as exc:
        return None, str(exc)
    except Exception:  # noqa: BLE001 - a broken fallback store is "no credential"
        logger.warning(
            "Credential fallback lookup failed for server=%s", server_config.name, exc_info=False
        )
        return None, None
    if credential is not None and not (credential.access_token or credential.api_key):
        return None, None
    return credential, None


def _precheck(
    credential: UserCredential | None,
    server_config: MCPServerConfig,
    missing_reason: str | None,
) -> tuple[ToolStatus, ToolError] | None:
    """Reject bad credentials before any network call. Order: missing,
    expired, provider mismatch. Returns None when the call may proceed."""
    server = server_config.name
    secret = (credential.access_token or credential.api_key) if credential else None
    if not secret or not secret.strip():
        status, error = errors.not_authenticated(server)
        if missing_reason:
            error.message = missing_reason
        return status, error

    if credential.expiry is not None and credential.expiry < time.time() + _EXPIRY_BUFFER_SECONDS:
        return errors.token_expired(server)

    if credential.provider and credential.auth_type == "oauth2":
        expected = server_config.auth_provider
        if not expected or expected.lower() != credential.provider.lower():
            return errors.provider_mismatch(server, expected, credential.provider)

    return None


def _effective_source(entry: CatalogEntry, call: ToolCall) -> ToolSource:
    """Resolve effective source: caller-declared wins over catalog."""
    try:
        return ToolSource(call.source.upper()) if call.source else entry.source
    except ValueError:
        return entry.source


def _make_sub_executor(
    catalog: "ToolCatalog",
    mcp_configs: dict,
    granted_permissions: list[str],
    timeout_seconds: int,
    breaker: "CircuitBreaker | None",
    user_id: "str | None",
    credential_manager: "CredentialManager | None",
    rest_settings: "RestExecutionSettings | None" = None,
    disabled_servers: frozenset[str] = frozenset(),
):
    """Return an async callable that executes a single tool call through the
    normal executor path.  Injected into ``batch_process`` at dispatch time so
    it can call other tools without importing executor directly.

    Signature of the returned callable::

        async def _executor(tool_name: str, tool_input: dict) -> dict
            # {"success": bool, "output": dict|None, "error": str|None}

    ``rest_settings`` is forwarded so that API-source tools called as
    ``nextTool`` inside a batch are routed correctly to the REST backend
    rather than falling through to the native runner.
    """
    _rest = rest_settings or get_rest_settings()

    async def _executor(tool_name: str, tool_input: dict) -> dict:
        entry = resolve(catalog, tool_name)
        tool_source = entry.source.value if entry is not None else "NATIVE"
        sub_call = ToolCall(
            tool_call_id=f"batch-{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            input=tool_input,
            source=tool_source,
        )
        result = await execute_call(
            call=sub_call,
            catalog=catalog,
            granted_permissions=granted_permissions,
            timeout_seconds=timeout_seconds,
            mcp_configs=mcp_configs,
            breaker=breaker,
            user_id=user_id,
            credential_manager=credential_manager,
            rest_settings=_rest,
            disabled_servers=disabled_servers,
        )

    _rest = rest_settings or get_rest_settings()

    async def _executor(tool_name: str, tool_input: dict) -> dict:
        # Resolve the source from the catalog so MCP/API tools dispatch correctly.
        # We must NOT hardcode source="NATIVE" on the sub-call — that would override
        # the catalog entry and send MCP/API tools to the native runner (causing
        # "no native tool registered as '...'" errors). Setting source="" lets
        # _effective_source fall through to entry.source from the catalog.
        entry = resolve(catalog, tool_name)
        tool_source = entry.source.value if entry is not None else "NATIVE"
        sub_call = ToolCall(
            tool_call_id=f"batch-{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            input=tool_input,
            source=tool_source,
        )
        result = await execute_call(
            call=sub_call,
            catalog=catalog,
            granted_permissions=granted_permissions,
            timeout_seconds=timeout_seconds,
            mcp_configs=mcp_configs,
            breaker=breaker,
            user_id=user_id,
            credential_manager=credential_manager,
            rest_settings=_rest,
            disabled_servers=disabled_servers,
        )
        if result.status == ToolStatus.SUCCESS:
            return {"success": True, "output": result.output}
        error_msg = result.error.message if result.error else f"status={result.status.value}"
        return {"success": False, "error": error_msg}

    return _executor


async def _dispatch(
    entry: CatalogEntry,
    call: ToolCall,
    mcp_configs: dict[str, MCPServerConfig],
    credential: UserCredential | None = None,
    catalog: "ToolCatalog | None" = None,
    user_id: str | None = None,
    credential_manager: "CredentialManager | None" = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    granted_permissions: list[str] | None = None,
    rest_settings: RestExecutionSettings | None = None,
    *,
    disabled_servers: frozenset[str] = frozenset(),
) -> dict:
    """Route to the execution backend for this entry's source."""
    effective_source = _effective_source(entry, call)

    if effective_source is ToolSource.API:
        settings = rest_settings or get_rest_settings()
        dag = _build_api_dag(entry, call)
        return await rest_client.start_workflow(settings, dag)

    if effective_source is ToolSource.MCP:
        if entry.server is None:
            raise MCPConfigError(f"MCP entry {entry.name!r} has no server")
        config = mcp_configs.get(entry.server)
        if config is None:
            raise MCPConfigError(
                f"MCP server {entry.server!r} was not reachable during discovery"
            )
        from ai_os.app.config.settings import get_settings as _get_settings
        if _get_settings().mcp_transport == "queue":
            from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.queue_client import (
                get_queue_client,
            )
            return await get_queue_client().call_tool(
                config, entry.target, call.input, credential,
                timeout_seconds=timeout_seconds,
            )
        return await mcp_client.call_tool(config, entry.target, call.input, credential)

    # NATIVE (default)
    # batch_process needs a sub-executor callback injected so it can call
    # other tools internally without importing executor directly (avoids
    # circular imports). The _executor kwarg is NOT part of the public schema
    # and is never supplied by the LLM — only injected here.
    if entry.target == "batch_process":
        if catalog is None:
            raise RuntimeError(
                "batch_process requires a ToolCatalog to be passed through _dispatch. "
                "This is an internal configuration error."
            )
        sub_executor = _make_sub_executor(
            catalog=catalog,
            mcp_configs=mcp_configs,
            granted_permissions=granted_permissions or [],
            timeout_seconds=timeout_seconds,
            breaker=None,             # sub-calls get their own fresh breaker state
            user_id=user_id,
            credential_manager=credential_manager,
            rest_settings=rest_settings or get_rest_settings(),
            disabled_servers=disabled_servers,
        )
        return await runner.invoke(entry.target, {**call.input, "_executor": sub_executor})

    return await runner.invoke(entry.target, call.input)


# These two keys decide WHICH action the backend runs
# (e.g. resource="message", operation="get").
# If dagConfig.defaultParameters sets them, they are LOCKED:
# the LLM cannot change them.
#
# SAME RULE as agent-runtime-service dag/dag_builder.py
# (_LOCKED_ACTION_KEYS / _remove_locked_action_keys). Keep both in sync.
_LOCKED_ACTION_KEYS = frozenset({"resource", "operation"})


def _remove_locked_action_keys(
    llm_input: dict[str, Any] | None,
    defaults: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    """Return a copy of the LLM's input without the locked keys.

    A key is locked only when BOTH are true:
      - it is "resource" or "operation" (any upper/lower case), and
      - ``defaults`` (dagConfig.defaultParameters) sets it.

    Example:
        defaults  = {"resource": "message", "operation": "get"}
        llm_input = {"message_id": "m1", "Operation": "delete"}
        returns     {"message_id": "m1"}   # "Operation" removed, warning logged

    If ``defaults`` has no "operation", the LLM's "operation" is kept.
    """
    if not llm_input:
        return {}

    # Locked keys for this tool, lower-case -> value from defaults.
    locked_values = {
        key.lower(): value
        for key, value in defaults.items()
        if key.lower() in _LOCKED_ACTION_KEYS
    }
    if not locked_values:
        return dict(llm_input)

    allowed_input: dict[str, Any] = {}
    for key, value in llm_input.items():
        is_locked = isinstance(key, str) and key.lower() in locked_values
        if not is_locked:
            allowed_input[key] = value
            continue

        # Drop it. Warn only if the LLM tried a DIFFERENT value.
        locked_value = locked_values[key.lower()]
        if value != locked_value:
            logger.warning(
                "Ignored LLM value %r=%r for tool '%s'. Locked to %r by dagConfig.",
                key, value, tool_name, locked_value,
            )
    return allowed_input


def _build_api_dag(entry: CatalogEntry, call: ToolCall) -> dict[str, Any]:
    """Build a trigger → action DAG for an API-source tool call.

    The backend's workflow-start API requires:
      - A trigger node of type "trigger" / value "AGENT" as the entry point.
      - The action node connected via an edge from the trigger.
      - Envelope fields: rediskey, triggerType, name, version, status,
        description, execution_input, nodes, edges, user_id.

    This mirrors ``services/agent-runtime-service/.../dag/dag_builder.py``
    ``build_single_action_dag()`` exactly — that function is the working
    reference for direct (non-batch) API tool calls. batch_process routes
    its sub-calls through this same path, so they now produce identical DAGs.
    """
    import uuid as _uuid

    dag_config: dict[str, Any] = entry.dag_config or {}

    trigger_key = str(_uuid.uuid4())
    action_key = str(_uuid.uuid4())

    trigger_node: dict[str, Any] = {
        "node_key": trigger_key,
        "nodeId": trigger_key,
        "name": "Agent",
        "type": "trigger",
        "value": "AGENT",
        "expressionExecutionName": "Agent",
        "parameters": {},
        "trigger_type": 6,
    }

    # Step 1: start with the defaults from dagConfig.
    # Step 2: add the LLM's values on top (they win),
    #         but first remove resource / operation from them (locked).
    defaults: dict[str, Any] = dict(dag_config.get("defaultParameters") or {})
    parameters: dict[str, Any] = {
        **defaults,
        **_remove_locked_action_keys(call.input, defaults, entry.name),
    }
    # Execution-request dagConfig sends snake_case "credential_id"; the
    # file-based tools.json catalog uses camelCase "credentialId". Accept
    # either, preferring the snake_case form since that's what the live
    # execution payload actually carries (see agent_runtime_engine.py's
    # same dual-key read for the REST path).
    credential_id = dag_config.get("credential_id", dag_config.get("credentialId"))
    if credential_id is not None:
        parameters["credential_id"] = credential_id

    action_node: dict[str, Any] = {
        "node_key": action_key,
        "nodeId": action_key,
        "name": dag_config.get("name") or entry.name,
        "type": "action",
        "value": dag_config.get("value") or entry.target,
        "expressionExecutionName": dag_config.get("expressionExecutionName") or entry.name,
        "parameters": parameters,
    }

    edge: dict[str, Any] = {
        "id": f"e-{trigger_key}-out-{action_key}",
        "from_node": trigger_key,
        "to_node": action_key,
        "source_handle": "out",
        "target_handle": "in",
        "connection_type": "main",
        "edge_type": "normal",
        "loop_node_id": None,
    }

    # tool_call_id carries a sub-call id (e.g. "batch-abc123") — use it as
    # a stable rediskey suffix so concurrent batch rows don't collide.
    rediskey = f"batch-{call.tool_call_id}"

    return {
        "rediskey": rediskey,
        "triggerType": "Trigger",
        "name": dag_config.get("name") or entry.name,
        "version": 1,
        "status": 0,
        "description": "Workflow execution",
        "execution_input": {
            "json": [{"chatInput": "", "sessionId": ""}],
            "binary_data": [],
        },
        "nodes": [trigger_node, action_node],
        "edges": [edge],
        "user_id": 3,
    }


def _result(
    call: ToolCall,
    status: ToolStatus,
    output: dict | None = None,
    error: ToolError | None = None,
    duration_ms: float = 0.0,
) -> ToolCallResult:
    return ToolCallResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        status=status,
        output=output,
        error=error,
        duration_ms=duration_ms,
    )


def aborted_result(call: ToolCall) -> ToolCallResult:
    """Result for a call that was never run because the batch aborted."""
    return _result(
        call,
        ToolStatus.SKIPPED,
        error=ToolError(
            code="BATCH_ABORTED",
            message="not executed because an earlier call failed and onFailure is ABORT",
        ),
    )