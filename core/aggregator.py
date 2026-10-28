"""Combine per-call results into the engine's output contract.

The aggregate status mirrors the kernel's ``ResponseStatus`` values so the adapter
maps it without a lookup table:

- every call succeeded          -> SUCCESS
- no call succeeded             -> FAILURE
- some succeeded, some did not  -> PARTIAL

A batch where everything was SKIPPED (for example, all tool names unknown)
produces FAILURE. Nothing was accomplished, so reporting success would be
misleading even though nothing technically errored.
"""

from ai_os.app.infrastructure.engines.tool_engine.domain.contracts import (
    ResultSummary,
    ToolCallResult,
    ToolExecutionResult,
)
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import AggregateStatus, ExecutionMode, ToolStatus

_COUNTER_FIELD: dict[ToolStatus, str] = {
    ToolStatus.SUCCESS: "success",
    ToolStatus.FAILED: "failed",
    ToolStatus.DENIED: "denied",
    ToolStatus.SKIPPED: "skipped",
    ToolStatus.TIMEOUT: "timeout",
    ToolStatus.APPROVAL_REQUIRED: "approval_required",
    ToolStatus.RATE_LIMITED: "rate_limited",
}


def summarise(results: list[ToolCallResult]) -> ResultSummary:
    counts: dict[str, int] = {field: 0 for field in set(_COUNTER_FIELD.values())}
    for result in results:
        field_name = _COUNTER_FIELD.get(result.status, "failed")
        counts[field_name] = counts.get(field_name, 0) + 1
    return ResultSummary(total=len(results), **counts)


def derive_status(summary: ResultSummary) -> AggregateStatus:
    if summary.total > 0 and summary.success == summary.total:
        return AggregateStatus.SUCCESS
    if summary.success == 0:
        return AggregateStatus.FAILURE
    return AggregateStatus.PARTIAL


def build(
    results: list[ToolCallResult],
    mode: ExecutionMode,
    total_duration_ms: float,
) -> ToolExecutionResult:
    summary = summarise(results)
    return ToolExecutionResult(
        status=derive_status(summary),
        execution_mode=mode,
        summary=summary,
        results=results,
        total_duration_ms=total_duration_ms,
    )
