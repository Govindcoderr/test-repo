"""Run a batch of tool calls sequentially or in parallel.

``ExecutionMode`` and ``OnFailure`` are independent, so all four combinations are
handled:

======================  ==========================================================
Combination             Behaviour
======================  ==========================================================
SEQUENTIAL + CONTINUE   run every call in order
SEQUENTIAL + ABORT      stop after the first non-success; rest are SKIPPED
PARALLEL + CONTINUE     start all at once, wait for all
PARALLEL + ABORT        start all at once; on the first non-success, cancel the
                        calls still in flight and mark them SKIPPED
======================  ==========================================================

SEQUENTIAL mode also threads completed outputs forward, which is what lets a call
reference an earlier one with ``$ref`` - see ``core/references.py``. PARALLEL mode
cannot, because every call starts before any has finished, so the engine rejects a
parallel batch containing references instead of resolving them to nothing.

PARALLEL + ABORT cannot un-start work that already began - that is inherent to
running things at the same time, not a shortcut. What it can do is stop calls that
have not finished, which is what cancellation gives us.

Results are always returned in the order the calls were requested, regardless of
the order they finished in, so the caller can map results back to its request.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import TYPE_CHECKING

from ai_os.app.infrastructure.engines.tool_engine.catalog.loader import ToolCatalog
from ai_os.app.infrastructure.engines.tool_engine.core.circuit_breaker import CircuitBreaker
from ai_os.app.infrastructure.engines.tool_engine.core.executor import aborted_result, execute_call
from ai_os.app.infrastructure.engines.tool_engine.domain.contracts import ToolCall, ToolCallResult
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import ExecutionMode, OnFailure, ToolStatus
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.config import MCPServerConfig

if TYPE_CHECKING:
    from ai_os.app.infrastructure.engines.tool_engine.infrastructure.credentials.credential_manager import (
        CredentialManager,
    )


async def run_batch(
    calls: list[ToolCall],
    catalog: ToolCatalog,
    granted_permissions: list[str],
    timeout_seconds: int,
    mode: ExecutionMode,
    on_failure: OnFailure,
    mcp_configs: dict[str, MCPServerConfig] | None = None,
    breaker: CircuitBreaker | None = None,
    user_id: str | None = None,
    credential_manager: "CredentialManager | None" = None,
    disabled_servers: frozenset[str] = frozenset(),
) -> tuple[list[ToolCallResult], float]:
    """Execute all calls. Returns results in request order plus wall-clock ms."""
    started = perf_counter()
    configs = mcp_configs or {}

    if mode is ExecutionMode.SEQUENTIAL:
        results = await _run_sequential(
            calls, catalog, granted_permissions, timeout_seconds,
            on_failure, configs, breaker, user_id, credential_manager,
            disabled_servers,
        )
    else:
        results = await _run_parallel(
            calls, catalog, granted_permissions, timeout_seconds,
            on_failure, configs, breaker, user_id, credential_manager,
            disabled_servers,
        )

    return results, round((perf_counter() - started) * 1000, 3)


async def _run_sequential(
    calls: list[ToolCall],
    catalog: ToolCatalog,
    granted_permissions: list[str],
    timeout_seconds: int,
    on_failure: OnFailure,
    mcp_configs: dict[str, MCPServerConfig],
    breaker: CircuitBreaker | None,
    user_id: str | None,
    credential_manager: "CredentialManager | None",
    disabled_servers: frozenset[str] = frozenset(),
) -> list[ToolCallResult]:
    results: list[ToolCallResult] = []
    outputs: dict[str, dict | None] = {}

    for index, call in enumerate(calls):
        result = await execute_call(
            call,
            catalog,
            granted_permissions,
            timeout_seconds,
            outputs=outputs,
            mcp_configs=mcp_configs,
            breaker=breaker,
            user_id=user_id,
            credential_manager=credential_manager,
            disabled_servers=disabled_servers,
        )
        results.append(result)
        outputs[call.tool_call_id] = result.output if result.status is ToolStatus.SUCCESS else None

        if on_failure is OnFailure.ABORT and result.status is not ToolStatus.SUCCESS:
            results.extend(aborted_result(c) for c in calls[index + 1:])
            break

    return results


async def _run_parallel(
    calls: list[ToolCall],
    catalog: ToolCatalog,
    granted_permissions: list[str],
    timeout_seconds: int,
    on_failure: OnFailure,
    mcp_configs: dict[str, MCPServerConfig],
    breaker: CircuitBreaker | None,
    user_id: str | None,
    credential_manager: "CredentialManager | None",
    disabled_servers: frozenset[str] = frozenset(),
) -> list[ToolCallResult]:
    async def indexed(position: int, call: ToolCall) -> tuple[int, ToolCallResult]:
        return position, await execute_call(
            call, catalog, granted_permissions, timeout_seconds,
            mcp_configs=mcp_configs, breaker=breaker,
            user_id=user_id, credential_manager=credential_manager,
            disabled_servers=disabled_servers,
        )

    tasks = {asyncio.create_task(indexed(position, call)) for position, call in enumerate(calls)}
    slots: list[ToolCallResult | None] = [None] * len(calls)
    pending = set(tasks)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        should_abort = False
        for task in done:
            position, result = task.result()
            slots[position] = result
            if on_failure is OnFailure.ABORT and result.status is not ToolStatus.SUCCESS:
                should_abort = True

        if should_abort and pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            pending = set()

    return [
        slot if slot is not None else aborted_result(calls[position])
        for position, slot in enumerate(slots)
    ]
