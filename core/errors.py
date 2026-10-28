"""Exception-to-status mapping, in one place.

Every failure inside the engine leaves as a status value, never as a raised
exception. The kernel logs ``PROVIDER_RESOLVED -> ENGINE_DISPATCH ->
ENGINE_RESPONSE`` for each engine call; an exception crossing the boundary breaks
that trace. Centralising the mapping here keeps the rule enforceable instead of
scattered across executors.

Mapping:

===============================  ==========  ==========================
Situation                        Status      Code
===============================  ==========  ==========================
tool ran, worked                 SUCCESS     -
tool not in catalog              SKIPPED     TOOL_NOT_FOUND
native target not registered     SKIPPED     TOOL_NOT_IMPLEMENTED
MCP server not configured        SKIPPED     MCP_NOT_CONFIGURED
a referenced call did not work   SKIPPED     DEPENDENCY_FAILED
circuit breaker is open          SKIPPED     CIRCUIT_OPEN (retryable)
caller lacks permission          DENIED      PERMISSION_DENIED
bad or missing tool input        FAILED      INVALID_INPUT
malformed $ref                   FAILED      INVALID_REFERENCE
MCP server unreachable           FAILED      MCP_UNAVAILABLE (retryable)
MCP tool failed on the server    FAILED      MCP_TOOL_ERROR
tool exceeded the timeout        TIMEOUT     TOOL_TIMEOUT (retryable)
tool raised anything else        FAILED      TOOL_ERROR
===============================  ==========  ==========================

Retryable is set only where a repeat attempt could plausibly succeed: a transport
fault or a timeout. A tool that ran and rejected the input will reject it again.
"""

import asyncio

from ai_os.app.infrastructure.engines.tool_engine.core.references import DependencyFailedError, InvalidReferenceError
from ai_os.app.infrastructure.engines.tool_engine.domain.contracts import ToolError
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import ToolStatus
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.client import MCPConnectionError, MCPToolError
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.config import MCPConfigError
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.native.runner import UnknownNativeToolError


def tool_not_found(name: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.SKIPPED, ToolError(
        code="TOOL_NOT_FOUND",
        message=f"tool {name!r} is not in the catalog or is disabled",
    )


def permission_denied(name: str, permission: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.DENIED, ToolError(
        code="PERMISSION_DENIED",
        message=f"tool {name!r} requires permission {permission!r}",
    )


def classify(exc: BaseException) -> tuple[ToolStatus, ToolError]:
    """Map an exception raised during execution onto a status and error.

    ``TimeoutError`` is checked first because ``asyncio.TimeoutError`` is an alias
    of the builtin from Python 3.11 onward, and a timeout is not a tool defect.
    """
    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return ToolStatus.TIMEOUT, ToolError(
            code="TOOL_TIMEOUT",
            message="tool exceeded the allotted timeout",
            retryable=True,
        )

    if isinstance(exc, UnknownNativeToolError):
        return ToolStatus.SKIPPED, ToolError(
            code="TOOL_NOT_IMPLEMENTED",
            message=str(exc),
        )

    if isinstance(exc, DependencyFailedError):
        # The referenced call did not succeed, so the value this call needs does not
        # exist. Running the tool with a hole in its input would be worse.
        return ToolStatus.SKIPPED, ToolError(code="DEPENDENCY_FAILED", message=str(exc))

    if isinstance(exc, InvalidReferenceError):
        return ToolStatus.FAILED, ToolError(code="INVALID_REFERENCE", message=str(exc))

    if isinstance(exc, MCPToolError):
        # The server ran the tool and reported a failure.
        return classify_mcp(exc, connection=False)

    if isinstance(exc, MCPConnectionError):
        # The MCP endpoint was unreachable or rejected the session.
        return classify_mcp(exc, connection=True)

    if isinstance(exc, MCPConfigError):
        # A deployment fault rather than a request fault, so SKIPPED: the tool was
        # never runnable, and retrying will not change that.
        return ToolStatus.SKIPPED, ToolError(code="MCP_NOT_CONFIGURED", message=str(exc))

    if isinstance(exc, TypeError):
        # Wrong or missing keyword arguments surface as TypeError when the tool
        # function is called, which is an input problem rather than a tool defect.
        return ToolStatus.FAILED, ToolError(
            code="INVALID_INPUT",
            message=f"input does not match the tool signature: {exc}",
        )

    return ToolStatus.FAILED, ToolError(
        code="TOOL_ERROR",
        message=f"{type(exc).__name__}: {exc}",
    )


# ── MCP failure classification ───────────────────────────────────────────────

# Only real outages move the circuit breaker. Auth problems, throttling and
# bad input are the caller's or the user's problem: tripping the breaker on
# them would block every other user of the same tool.
BREAKER_COUNTED_CODES: frozenset[str] = frozenset({"MCP_UNAVAILABLE", "TOOL_TIMEOUT"})

_MAX_RETRY_AFTER_SECONDS = 86_400

# Structured errorCode values an MCP server may report, mapped to
# (status, code, retryable). Unknown codes fall through to httpStatus.
_KNOWN_MCP_CODES: dict[str, tuple[ToolStatus, str, bool]] = {
    "AUTH_FAILED": (ToolStatus.DENIED, "AUTH_FAILED", False),
    "NOT_AUTHENTICATED": (ToolStatus.DENIED, "AUTH_FAILED", False),
    "RATE_LIMITED": (ToolStatus.RATE_LIMITED, "RATE_LIMITED", True),
    "NOT_SUPPORTED_FOR_ACCOUNT_TYPE": (ToolStatus.FAILED, "NOT_SUPPORTED_FOR_ACCOUNT_TYPE", False),
    "INVALID_INPUT": (ToolStatus.FAILED, "INVALID_INPUT", False),
    "MCP_UNAVAILABLE": (ToolStatus.FAILED, "MCP_UNAVAILABLE", True),
    "UPSTREAM_UNAVAILABLE": (ToolStatus.FAILED, "MCP_UNAVAILABLE", True),
    "MCP_TOOL_ERROR": (ToolStatus.FAILED, "MCP_TOOL_ERROR", False),
}

_NOT_SUPPORTED_MESSAGE = (
    "This tool is not available for the user's Microsoft account type "
    "(work/school vs personal)."
)


def counts_toward_breaker(error: ToolError) -> bool:
    """Whether a failure with this error should increment the breaker."""
    return error.code in BREAKER_COUNTED_CODES


def classify_mcp(exc, *, connection: bool) -> tuple[ToolStatus, ToolError]:
    """Classify an MCP failure.

    Structured fields win: ``errorCode`` first, then ``httpStatus``. Text
    matching is used only when the server sent neither. The result depends
    only on the exception type, ``errorCode``, ``httpStatus`` and message, so
    the same input always classifies the same way.
    """
    code = getattr(exc, "error_code", None)
    status = getattr(exc, "http_status", None)
    retry_after = _valid_retry_after(getattr(exc, "retry_after_seconds", None))
    message = str(exc)

    if code in _KNOWN_MCP_CODES:
        tool_status, out_code, retryable = _KNOWN_MCP_CODES[code]
        return _mcp_error(tool_status, out_code, retryable, message, retry_after)

    if isinstance(status, int):
        return _from_http_status(status, message, retry_after)

    if code:
        # A structured code we do not know: a tool-level failure.
        return ToolStatus.FAILED, ToolError(code="MCP_TOOL_ERROR", message=message)

    lowered = message.lower()
    if any(t in lowered for t in ("401", "403", "unauthorized", "forbidden")):
        return _mcp_error(ToolStatus.DENIED, "AUTH_FAILED", False, message, None)
    if any(t in lowered for t in ("429", "rate limit", "too many requests")):
        return _mcp_error(ToolStatus.RATE_LIMITED, "RATE_LIMITED", True, message, None)
    if connection:
        return ToolStatus.FAILED, ToolError(code="MCP_UNAVAILABLE", message=message, retryable=True)
    return ToolStatus.FAILED, ToolError(code="MCP_TOOL_ERROR", message=message)


def _from_http_status(
    status: int, message: str, retry_after: int | None
) -> tuple[ToolStatus, ToolError]:
    if status in (401, 403):
        return _mcp_error(ToolStatus.DENIED, "AUTH_FAILED", False, message, None)
    if status == 429:
        return _mcp_error(ToolStatus.RATE_LIMITED, "RATE_LIMITED", True, message, retry_after)
    if 500 <= status <= 599:
        return ToolStatus.FAILED, ToolError(code="MCP_UNAVAILABLE", message=message, retryable=True)
    return ToolStatus.FAILED, ToolError(code="MCP_TOOL_ERROR", message=message)


def _mcp_error(
    status: ToolStatus,
    code: str,
    retryable: bool,
    message: str,
    retry_after: int | None,
) -> tuple[ToolStatus, ToolError]:
    if code == "NOT_SUPPORTED_FOR_ACCOUNT_TYPE":
        message = _NOT_SUPPORTED_MESSAGE
    error = ToolError(code=code, message=message, retryable=retryable)
    if code == "RATE_LIMITED" and retry_after is not None:
        error.retry_after_seconds = retry_after
    return status, error


def _valid_retry_after(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = int(value)
    if 0 <= seconds <= _MAX_RETRY_AFTER_SECONDS:
        return seconds
    return None


# ── Credential pre-check results ─────────────────────────────────────────────
# None of these reach the MCP server and none move the circuit breaker.

def not_authenticated(server: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.DENIED, ToolError(
        code="NOT_AUTHENTICATED",
        message=f"No access token for MCP server {server!r}. The user must connect the account.",
    )


def token_expired(server: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.DENIED, ToolError(
        code="TOKEN_EXPIRED",
        message=f"Access token for MCP server {server!r} is expired or expires within 30s.",
    )


def provider_mismatch(server: str, expected: str | None, got: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.DENIED, ToolError(
        code="TOKEN_PROVIDER_MISMATCH",
        message=(
            f"Credential provider {got!r} does not match MCP server {server!r} "
            f"(authProvider={expected!r})."
        ),
    )


def invalid_credential(code: str, server: str, detail: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.DENIED, ToolError(
        code=code,
        message=f"Invalid credential config for MCP server {server!r}: {detail}",
    )


def provider_disabled(server: str) -> tuple[ToolStatus, ToolError]:
    return ToolStatus.SKIPPED, ToolError(
        code="PROVIDER_DISABLED",
        message=f"MCP server {server!r} is disabled by its provider flag.",
    )


def circuit_open(name: str, key: str) -> tuple[ToolStatus, ToolError]:
    """The circuit breaker is OPEN for this tool — fail fast without dispatch."""
    return ToolStatus.SKIPPED, ToolError(
        code="CIRCUIT_OPEN",
        message=(
            f"Circuit breaker is OPEN for tool {name!r} (key={key!r}). "
            "The tool has been failing repeatedly; calls are paused until it recovers."
        ),
        retryable=True,
    )
