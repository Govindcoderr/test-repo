"""Per-request waiter: fold events for a ``workflow_definition_id`` into a result.

Called by ``engine.run_rest`` right after the REST backend's ack. The
tracker is stateless (no in-process map of pending wdids) — all state
lives in Redis, which is what makes the "event arrived on another pod"
case work.

Correlation rule
----------------
Events are picked by the ``workflow_definition_id`` field they carry. Live
captures show that field holds the DAG's ``rediskey`` (NOT the numeric id in
the REST ack), so ``engine.run_rest`` passes the rediskey as ``wdid`` here.
Never key by ``nodeId``.

Folding rules
-------------
  * The current status monotonically moves forward: once we see
    completed/failed/cancelled we never downgrade. This makes late
    duplicate events harmless.
  * Nodes are collected into a dict keyed by ``(sequence, run_count)``.
    Later events for the same key replace earlier ones — in practice that
    is a ``running → completed`` transition.
  * The workflow-level output is the ``output`` of the node with the
    highest sequence that has a non-empty output. The summary event's
    empty output never wins because it has no sequence.
  * Timeouts return the partial fold with status ``timeout`` and whatever
    ``last_status`` we saw. The result is still cached in Redis so the
    caller can retrieve it if they re-fire with the same wdid.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.models import (
    WorkflowStatus,
    event_terminates_workflow,
    is_terminal_status,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.store import (
    WorkflowEventStore,
)

logger = logging.getLogger("tool-engine.workflow-status.tracker")

# Upper bound for a single XREAD BLOCK. MUST stay well below the Redis
# client's socket_timeout (runtime.py): if the server blocks longer than the
# socket allows, redis-py raises "Timeout reading from ..." even though
# nothing is wrong. Short blocks also let us re-check the deadline often.
_MAX_BLOCK_MS = 5_000


def _is_empty(value: Any) -> bool:
    """Treat None, {}, [] and empty strings as "no output"."""
    if value is None:
        return True
    if isinstance(value, (dict, list, str)) and len(value) == 0:
        return True
    return False


class WorkflowTracker:
    """Fold Redis-buffered events for a single workflow run."""

    def __init__(self, store: WorkflowEventStore) -> None:
        self._store = store

    async def wait(
        self,
        wdid: str,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Block until the workflow terminates or the budget expires.

        The returned dict is the final folded result::

            {
              "workflowStatus": "completed|failed|cancelled|timeout",
              "output": <node output or None>,
              "nodes": [ { sequence, run_count, status, input, output, ... } ],
              "error": "<str or None>",
              "last_status": "<last known workflow or node status>",
              "events_seen": <int>,
              "elapsed_seconds": <float>,
            }

        Never raises for a timeout — the caller gets a ``timeout`` status
        back. Other unexpected exceptions bubble up (they're bugs).
        """
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        last_id = "0"
        nodes: dict[tuple[Any, Any], dict[str, Any]] = {}
        last_status: str | None = None
        terminal: WorkflowStatus | None = None
        error: str | None = None
        error_code: str | None = None
        events_seen = 0
        started = time.monotonic()

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            block_ms = min(_MAX_BLOCK_MS, int(remaining * 1000))
            # xread must block at least a tick, otherwise it degrades into a
            # busy-poll if the stream is empty.
            block_ms = max(100, block_ms)

            try:
                batch = await asyncio.wait_for(
                    self._store.read_events_blocking(
                        wdid, last_id=last_id, block_ms=block_ms
                    ),
                    timeout=remaining + 1.0,
                )
            except asyncio.TimeoutError:
                break

            if not batch:
                continue

            for entry_id, parsed in batch:
                last_id = entry_id
                events_seen += 1
                _fold_event(parsed, nodes)

                ws = parsed.get("workflow_status")
                if isinstance(ws, str):
                    last_status = ws
                node = parsed.get("node") or {}
                ns = node.get("status")
                if isinstance(ns, str):
                    last_status = ns  # node status is more recent per event

                if not error and parsed.get("error"):
                    error = parsed["error"]
                if not error_code and parsed.get("error_code"):
                    error_code = parsed["error_code"]

                maybe_terminal = event_terminates_workflow(parsed)
                if maybe_terminal is not None and not is_terminal_status(terminal):
                    terminal = maybe_terminal

            if is_terminal_status(terminal):
                break

        final_status = terminal or WorkflowStatus.TIMEOUT
        result = _build_result(
            wdid=wdid,
            final_status=final_status,
            nodes=nodes,
            last_status=last_status,
            error=error,
            error_code=error_code,
            events_seen=events_seen,
            elapsed=time.monotonic() - started,
        )
        print(
            f"[JETSTREAM] 3. DONE     rediskey={wdid}  status={result['workflowStatus']}  "
            f"events={events_seen}  time={result['elapsed_seconds']}s",
            flush=True,
        )
        # Cache the outcome so a retry on the same wdid is instant and so
        # ops can read it after the fact.
        await self._store.store_result(wdid, result)
        await self._store.clear_pending(wdid)
        return result


# ─── Pure helpers (testable, no I/O) ──────────────────────────────────────


def _fold_event(parsed: dict[str, Any], nodes: dict[tuple[Any, Any], dict[str, Any]]) -> None:
    """Update the running node map in place.

    Summary events (no sequence) carry no useful per-node payload — we only
    use them for terminal detection in the caller.
    """
    if parsed.get("is_summary"):
        return
    node = parsed.get("node") or {}
    seq = node.get("sequence")
    run = node.get("run_count")
    if seq is None:
        return
    key = (seq, run)
    existing = nodes.get(key, {})
    # Later writes win per-field, but we preserve the earliest ``started_at``
    # and the latest non-empty ``output`` — mirrors how a human would read
    # the sequence.
    merged = {**existing, **{k: v for k, v in node.items() if v is not None or k not in existing}}
    if not _is_empty(existing.get("started_at")) and _is_empty(node.get("started_at")):
        merged["started_at"] = existing["started_at"]
    if _is_empty(node.get("output")) and not _is_empty(existing.get("output")):
        merged["output"] = existing["output"]
    nodes[key] = merged


def _build_result(
    *,
    wdid: str,
    final_status: WorkflowStatus,
    nodes: dict[tuple[Any, Any], dict[str, Any]],
    last_status: str | None,
    error: str | None,
    events_seen: int,
    elapsed: float,
    error_code: str | None = None,
) -> dict[str, Any]:
    # Sort by sequence (treat missing as -1 so trigger-less future shapes
    # still produce a stable order).
    ordered = [n for _, n in sorted(nodes.items(), key=lambda kv: (kv[0][0] or -1, kv[0][1] or 0))]

    # Workflow output = output of the highest-sequence node that has one.
    output: Any = None
    for node in reversed(ordered):
        if not _is_empty(node.get("output")):
            output = node.get("output")
            break

    return {
        "workflowStatus": final_status.value,
        "workflowDefinitionId": wdid,
        "output": output,
        "nodes": ordered,
        "error": error,
        "code": error_code,
        "last_status": last_status,
        "events_seen": events_seen,
        "elapsed_seconds": round(elapsed, 3),
    }
