"""The Tool Engine's own request and result contracts.

These are the two free-form dicts the kernel contract leaves to us:

- ``ToolExecutionPayload`` goes inside ``KernelRequest.payload``
- ``ToolExecutionResult`` goes inside ``KernelResponse.result``

Everything around them (requestId, executionId, correlationId, timeout,
retryPolicy, top-level status) belongs to the kernel and is not redefined here.

Field naming follows the kernel repo's own pattern: snake_case Python attributes
with camelCase aliases, plus ``populate_by_name`` so both spellings parse. Dump
with ``by_alias=True`` when handing data to the kernel.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_os.app.infrastructure.engines.tool_engine.domain.enums import (
    AggregateStatus,
    ExecutionMode,
    OnFailure,
    ToolStatus,
)


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ToolCall(BaseModel):
    """One tool invocation requested by the caller."""

    model_config = ConfigDict(populate_by_name=True)

    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    input: dict[str, Any] = Field(default_factory=dict)
    # Execution source — tells the executor which dispatch path to use.
    # Matches ToolConfiguration.source: "NATIVE" | "MCP" | "API".
    # Defaults to "NATIVE" so existing calls without this field keep working.
    source: str = Field(default="NATIVE", alias="source")
    # Per-tool configuration from allowedTools[*].configuration.
    # Contains user_email, credentials (client_id/secret/token_uri),
    # and tokens (access_token/refresh_token/expiry).
    # Injected by Agent Runtime from the execution request's allowedTools config.
    # The executor reads this to build credentials — LLM never sees this field.
    tool_config: dict[str, Any] | None = Field(default=None, alias="toolConfig")


class ToolExecutionPayload(_Base):
    """Input contract: what the caller puts in ``KernelRequest.payload``.

    ``execution_mode`` and ``on_failure`` are separate on purpose. How to run
    (one at a time vs all at once) and what to do on failure (keep going vs stop)
    are independent choices, and all four combinations are valid.

    ``permissions`` carries the caller's granted permissions. The kernel's
    ``AuthorizationService.authorize()`` only decides at capability level
    (``tool.execute``), so per-tool enforcement has to happen here.

    ``user_id`` identifies the platform user on whose behalf the tools run.
    Required for MCP tools that need OAuth2 or API-key credentials.  Optional
    for NATIVE tools (ignored) and for MCP tools with ``auth_type == "none"``.
    The kernel / Agent Runtime injects this from the top-level execution
    request's ``userId`` field.

    There is no ``timeout`` field: ``KernelRequest.timeout`` already carries it
    (default 30s) and duplicating it would create two sources of truth.
    """

    tools: list[ToolCall] = Field(min_length=1)
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL, alias="executionMode")
    on_failure: OnFailure = Field(default=OnFailure.CONTINUE, alias="onFailure")
    permissions: list[str] = Field(default_factory=list)
    user_id: str | None = Field(default=None, alias="userId")
    """Platform user identity. Injected by the kernel from the top-level
    execution request's ``userId`` field. Used by the executor to fetch
    per-user credentials for MCP tool calls."""


class ToolError(_Base):
    """Why a tool call did not succeed."""

    code: str
    message: str
    retryable: bool = False
    retry_after_seconds: int | None = Field(default=None, alias="retryAfterSeconds")
    """Seconds to wait before retrying, when the upstream said so (HTTP 429)."""
    stage: str | None = None
    """The pipeline stage that refused the call, when applicable.
    None for tool-level failures. Populated from the Gateway's error code
    when the security pipeline rejected the call at a specific stage
    (e.g. AUTHORIZATION, RATE_LIMIT, INPUT_VALIDATION).
    """


class ToolCallResult(_Base):
    """Outcome of one tool call.

    ``duration_ms`` is always populated, including on failure and timeout, because
    the kernel aggregates tool time into ``executionMetrics.toolTime``.
    """

    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    status: ToolStatus
    output: dict[str, Any] | None = None
    error: ToolError | None = None
    duration_ms: float = Field(default=0.0, alias="durationMs")
    # Enrichment fields — populated when dispatching through the Tool Gateway.
    # Absent (None) when using direct native/MCP dispatch.
    tool_version: str | None = Field(default=None, alias="toolVersion")
    """The version of the tool that executed this call, as reported by the Gateway."""
    approval_id: str | None = Field(default=None, alias="approvalId")
    """Approval request ID when the Gateway suspended this call pending human review."""
    gateway_request_id: str | None = Field(default=None, alias="gatewayRequestId")
    """The Gateway's own per-call request ID. Allows joining this result to the
    Gateway's audit trail for debugging and observability."""


class ResultSummary(_Base):
    """Per-status counts across all calls in one batch."""

    total: int = 0
    success: int = 0
    failed: int = 0
    denied: int = 0
    skipped: int = 0
    timeout: int = 0
    approval_required: int = Field(default=0, alias="approvalRequired")
    rate_limited: int = Field(default=0, alias="rateLimited")


class ToolExecutionResult(_Base):
    """Output contract: what the engine puts in ``KernelResponse.result``.

    ``status`` mirrors the kernel's ``ResponseStatus`` values so the adapter maps
    it without a lookup table.
    """

    status: AggregateStatus
    execution_mode: ExecutionMode = Field(alias="executionMode")
    summary: ResultSummary
    results: list[ToolCallResult] = Field(default_factory=list)
    total_duration_ms: float = Field(default=0.0, alias="totalDurationMs")
    error: ToolError | None = None
    """Batch-level failure, set only when the payload itself could not be used.
    Per-call failures live on the individual results instead."""
