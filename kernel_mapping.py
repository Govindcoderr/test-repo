"""Pure translation between the engine's result and the kernel's response fields.

Kept free of ``ai_os`` imports on purpose so this logic is testable without the
kernel repo on the path. ``kernel_adapter.py`` does the actual wiring.

The mapping is deliberately thin. ``AggregateStatus`` was defined with the same
member names as the kernel's ``ResponseStatus``, so status translation is a
value passthrough rather than a lookup table.
"""

from ai_os.app.infrastructure.engines.tool_engine.domain.contracts import ToolExecutionResult
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import AggregateStatus, ToolStatus


def response_status(result: ToolExecutionResult) -> str:
    """Kernel ``ResponseStatus`` value for this batch.

    Note that a batch consisting entirely of timeouts still maps to FAILURE rather
    than the kernel's TIMEOUT status. TIMEOUT at the kernel level means the engine
    call itself timed out, which is a different event from a tool inside a
    successful engine call timing out.
    """
    return result.status.value


def response_metrics(result: ToolExecutionResult) -> dict[str, float | int]:
    """Metrics for ``KernelResponse.metrics``.

    The kernel aggregates tool time and call counts into
    ``executionMetrics.toolTime`` and ``totalToolCalls``, so both are reported
    here. ``toolTimeMs`` is the sum of individual call durations, which for a
    parallel batch exceeds wall-clock time; ``wallClockMs`` is reported separately
    so neither number has to be inferred.
    """
    return {
        "toolTimeMs": round(sum(r.duration_ms for r in result.results), 3),
        "wallClockMs": result.total_duration_ms,
        "totalToolCalls": result.summary.total,
        "successfulToolCalls": result.summary.success,
    }


def response_warnings(result: ToolExecutionResult) -> list[str]:
    """Human-readable warnings for the non-success calls in a partial batch."""
    if result.status is AggregateStatus.SUCCESS:
        return []

    warnings: list[str] = []
    for call_result in result.results:
        if call_result.status is ToolStatus.SUCCESS:
            continue
        detail = call_result.error.message if call_result.error else "no detail"
        warnings.append(
            f"{call_result.tool_name} [{call_result.tool_call_id}] "
            f"{call_result.status.value}: {detail}"
        )
    return warnings


def response_errors(result: ToolExecutionResult) -> list[dict[str, object]]:
    """Error entries for ``KernelResponse.errors``.

    Populated only when the whole batch failed. In a partial batch the failures are
    already visible in ``result`` and repeating them as top-level errors would
    misrepresent a partly successful call as a failed one.
    """
    if result.status is not AggregateStatus.FAILURE:
        return []

    if result.error is not None:
        return [
            {
                "code": result.error.code,
                "message": result.error.message,
                "retryable": result.error.retryable,
            }
        ]

    return [
        {
            "code": r.error.code if r.error else "TOOL_ERROR",
            "message": f"{r.tool_name}: {r.error.message if r.error else 'no detail'}",
            "retryable": r.error.retryable if r.error else False,
        }
        for r in result.results
        if r.status is not ToolStatus.SUCCESS
    ]
