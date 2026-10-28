"""Domain types and parsing helpers for workflow-status events.

Scope
-----
This module is intentionally pure Python: no NATS, no Redis, no I/O. That
keeps the parse/dedup/terminal-detection logic unit-testable against the
live-captured event shape without spinning up any infrastructure.

Event shape (observed, live sample)
-----------------------------------
Every event on ``dev.initial.workflow.execution.status.update`` looks like::

    {
      "type": "executionStateUpdate",
      "ws_connection_id": "<uuid>",
      "workflow_definition_id": "<uuid>",   # correlation key
      "payload": {
        "workflow_definition_id": "<uuid>", # same, duplicated for consumers
                                            # that only read payload
        "status": "running" | "completed" | "failed" | "cancelled",
        "node": {
          "nodeId": "<uuid>",
          "sequence": <int?>,               # absent on summary events
          "run_count": <int?>,
          "input": {...},
          "output": {...} | null,
          "started_at": "<iso?>",
          "completed_at": "<iso?>",
          "status": "running" | "completed" | "failed" | "cancelled"
        }
      }
    }

Important: ``payload.status`` is the **node's** status, not the workflow's
— the trigger node reports ``completed`` while the workflow is still
running. The workflow is only finished when:
  * the node is ``failed`` or ``cancelled``, OR
  * the summary event arrives (``sequence`` is absent, status ``completed``).

Correlation is **by ``workflow_definition_id`` only**. Nothing in this
module uses ``nodeId``, ``rediskey`` or ``ws_connection_id`` for matching.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

# Keys we accept for the workflow identifier on both the event and the
# backend's REST acknowledgement. We check top-level first, then common
# wrappers. The order matters — we return the first one that resolves to a
# non-empty string.
WORKFLOW_DEFINITION_ID_KEYS: tuple[str, ...] = (
    "workflow_definition_id",
    # "workflowDefinitionId",
)

_ACK_WRAPPER_KEYS: tuple[str, ...] = ("data", "result", "payload", "response", "details")


class WorkflowStatus(str, Enum):
    """Lifecycle states we expose to the caller.

    ``queued`` is a Tool-Engine-internal state used only when the backend
    accepted the DAG but the NATS pipeline is disabled or degraded. The
    backend itself never sends it.

    ``timeout`` is set by the tracker when the per-call budget elapses with
    no terminal event. The backend never sends it either.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


_TERMINAL: frozenset[WorkflowStatus] = frozenset(
    {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED, WorkflowStatus.TIMEOUT}
)


def is_terminal_status(status: WorkflowStatus | str | None) -> bool:
    """Is this a final status that must not be overwritten by later events?"""
    if status is None:
        return False
    try:
        return WorkflowStatus(status) in _TERMINAL
    except ValueError:
        return False


# ─── Extraction helpers ────────────────────────────────────────────────────


def _first_string(d: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """First non-empty id under ``keys``, always returned as ``str``.

    The backend ack sends ``workflow_definition_id`` as a JSON number
    (``2815``) while events may carry it as a string. Both are normalised
    to the same string so Redis keys (``wf:events:{wdid}``) line up between
    the waiter and the JetStream consumer. ``bool`` is excluded on purpose:
    it is an ``int`` subclass and ``True`` is never a valid id.
    """
    for k in keys:
        v = d.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_workflow_definition_id(obj: Any) -> str | None:
    """Pull ``workflow_definition_id`` out of a nested dict.

    Checks the top level, every known key alias, and the handful of
    wrapper keys the backend sometimes nests the real response under
    (``data``, ``result``, ``payload``, ``response``). Returns the first
    non-empty string match, or ``None``.
    """
    if not isinstance(obj, dict):
        return None
    found = _first_string(obj, WORKFLOW_DEFINITION_ID_KEYS)
    if found:
        return found
    for wrapper in _ACK_WRAPPER_KEYS:
        inner = obj.get(wrapper)
        if isinstance(inner, dict):
            found = extract_workflow_definition_id(inner)
            if found:
                return found
    return None


def extract_rediskey(obj: Any) -> str | None:
    """Best-effort rediskey lookup, used only for logging/trace."""
    if not isinstance(obj, dict):
        return None
    for key in ("rediskey", "redisKey", "redis_key"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for wrapper in _ACK_WRAPPER_KEYS:
        inner = obj.get(wrapper)
        if isinstance(inner, dict):
            found = extract_rediskey(inner)
            if found:
                return found
    return None


# ─── Event parsing / normalisation ─────────────────────────────────────────


def parse_event(raw: Any) -> dict[str, Any] | None:
    """Flatten the on-wire event shape into something easy to fold.

    Returns ``None`` for anything we cannot safely consume (not a dict, no
    workflow id). Callers must still ack malformed messages so they don't
    redeliver forever — the caller does that; we only decide "usable or not".

    The output dict has a stable schema::

        {
          "workflow_definition_id": "<str>",
          "workflow_status": "running|completed|failed|cancelled|None",
          "node": {                       # may be {} if the event has no node
            "nodeId": str|None,
            "sequence": int|None,         # None marks the summary event
            "run_count": int|None,
            "status": str|None,
            "input": Any,
            "output": Any,
            "started_at": str|None,
            "completed_at": str|None,
          },
          "is_summary": bool,             # sequence is None/missing
          "error": str|None,              # best-effort scrape
          "raw": <original dict>
        }
    """
    if not isinstance(raw, dict):
        return None

    wdid = extract_workflow_definition_id(raw)
    if not wdid:
        return None

    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    node_src = payload.get("node") if isinstance(payload.get("node"), dict) else {}

    sequence = node_src.get("sequence")
    is_summary = sequence is None  # see docstring: summary carries no sequence
    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    node_status = node_src.get("status") if isinstance(node_src.get("status"), str) else None

    # Error scrape — the live sample doesn't ship one yet, but we leave the
    # hooks in place so a future failed/cancelled event is surfaced to the
    # caller without a code change.
    error: str | None = None
    for candidate in (payload.get("error"), node_src.get("error"), raw.get("error")):
        if isinstance(candidate, str) and candidate.strip():
            error = candidate.strip()
            break
        if isinstance(candidate, dict):
            msg = candidate.get("message") or candidate.get("detail")
            if isinstance(msg, str) and msg.strip():
                error = msg.strip()
                break

    # A failed node reports its reason inside ``node.output`` (observed live:
    # ``{"error": true, "error_code": "NODE_EXECUTION_FAILED", "message": ...}``),
    # not in any of the ``error`` fields above. Without this fallback the real
    # cause (e.g. a Google 401) is lost and the caller sees a generic failure.
    error_code: str | None = None
    failed = FAILED_VALUE in (status, node_status)
    node_output = node_src.get("output")
    if failed and isinstance(node_output, dict):
        code = node_output.get("error_code")
        if isinstance(code, str) and code.strip():
            error_code = code.strip()
        if not error:
            msg = node_output.get("message")
            if isinstance(msg, str) and msg.strip():
                error = _with_auth_hint(msg.strip())
            elif isinstance(node_output.get("error"), str) and node_output["error"].strip():
                error = _with_auth_hint(node_output["error"].strip())

    return {
        "workflow_definition_id": wdid,
        "workflow_status": status,
        "node": {
            "nodeId": node_src.get("nodeId"),
            "sequence": sequence,
            "run_count": node_src.get("run_count"),
            "status": node_status,
            "input": node_src.get("input"),
            "output": node_src.get("output"),
            "started_at": node_src.get("started_at"),
            "completed_at": node_src.get("completed_at"),
        },
        "is_summary": is_summary,
        "error": error,
        "error_code": error_code,
        "raw": raw,
    }


FAILED_VALUE = "failed"

_AUTH_FAILURE_MARKERS: tuple[str, ...] = (
    "401",
    "unauthenticated",
    "invalid authentication credentials",
    "invalid credentials",
)


def _with_auth_hint(message: str) -> str:
    """Append a plain-language hint when the failure is an auth problem."""
    lowered = message.lower()
    if any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
        return (
            f"{message} -- The connected account's credential looks expired or "
            "invalid; reconnect it and try again."
        )
    return message


# ─── Workflow-level terminal detection ─────────────────────────────────────


def event_terminates_workflow(parsed: dict[str, Any]) -> WorkflowStatus | None:
    """Return the terminal workflow status if this event ends the run.

    Rules (derived from the live sample):
      * any event with ``failed`` or ``cancelled`` is terminal
      * the summary event (``is_summary``) carries the workflow status —
        if that is ``completed`` the run is done
      * a non-summary event with ``completed`` is a node finishing, not the
        workflow — do NOT terminate on it
    """
    status = parsed.get("workflow_status")
    if status == WorkflowStatus.FAILED.value:
        return WorkflowStatus.FAILED
    if status == WorkflowStatus.CANCELLED.value:
        return WorkflowStatus.CANCELLED

    node_status = (parsed.get("node") or {}).get("status")
    if node_status == WorkflowStatus.FAILED.value:
        return WorkflowStatus.FAILED
    if node_status == WorkflowStatus.CANCELLED.value:
        return WorkflowStatus.CANCELLED

    if parsed.get("is_summary") and status == WorkflowStatus.COMPLETED.value:
        return WorkflowStatus.COMPLETED
    return None


# ─── Dedup hash ───────────────────────────────────────────────────────────


def event_fingerprint(parsed: dict[str, Any]) -> str:
    """Stable hash of the fields that identify a logical event.

    Used by the store to drop NATS redeliveries and backend re-publishes.
    Any two events whose ``(sequence, run_count, workflow_status,
    node_status, completed_at)`` tuples match are treated as the same event
    — status rules in the tracker make a false positive harmless (we never
    overwrite a terminal state anyway).
    """
    node = parsed.get("node") or {}
    parts = (
        str(node.get("sequence")),
        str(node.get("run_count")),
        str(parsed.get("workflow_status")),
        str(node.get("status")),
        str(node.get("completed_at")),
        "summary" if parsed.get("is_summary") else "node",
    )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
