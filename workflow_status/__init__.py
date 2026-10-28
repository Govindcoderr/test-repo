"""NATS JetStream workflow-execution-status subsystem.

The REST backend's POST /workflowInit/.../start only *queues* a workflow and
returns a short acknowledgement (``rediskey`` + ``workflow_definition_id``).
The real outcome — ``running`` / ``completed`` / ``failed`` / ``cancelled``,
plus every node's input/output — arrives *out of band* on a JetStream subject
(``dev.initial.workflow.execution.status.update``).

This package bridges the two:

    REST POST ─ack─▶ run_rest ──waits on──▶ Redis stream ◀─writes── JS consumer
                                                                     ▲
                                                              ◀──── NATS JetStream

Why a Redis stream between the two:

  * The JS consumer and the request handler almost never share a pod — a
    durable JetStream consumer load-balances events across replicas. Redis
    is the only cross-replica meeting point we already have.
  * The whole run can finish in <500 ms, publishing all status events before
    the REST POST's HTTP ack even returns. Buffering in Redis means the
    waiter can read from the start and never miss an event.
  * A pod restart mid-wait would otherwise lose the result; Redis survives
    it.

Correlation is **by ``workflow_definition_id`` only** — not by rediskey, not
by nodeId. The ack must carry it; the event's top-level or
``payload.workflow_definition_id`` provides it on the receive side.

Public surface (lifespan + engine use):

    * ``get_settings()``                — cached config
    * ``WorkflowStatusRuntime``         — holds the live Redis client,
                                           consumer task and tracker
    * ``get_runtime()`` / ``set_runtime()``  — process-wide singleton,
                                           main.py's lifespan wires it,
                                           ``engine.run_rest`` reads it
    * ``WorkflowStatus`` (enum), ``WorkflowOutcome`` (TypedDict-ish dict)
"""

from __future__ import annotations

from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.config import (
    WorkflowStatusSettings,
    get_settings,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.models import (
    WORKFLOW_DEFINITION_ID_KEYS,
    WorkflowStatus,
    extract_workflow_definition_id,
    is_terminal_status,
    parse_event,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.runtime import (
    WorkflowStatusRuntime,
    build_runtime,
    get_runtime,
    set_runtime,
)

__all__ = [
    "WORKFLOW_DEFINITION_ID_KEYS",
    "WorkflowStatus",
    "WorkflowStatusRuntime",
    "WorkflowStatusSettings",
    "build_runtime",
    "extract_workflow_definition_id",
    "get_runtime",
    "get_settings",
    "is_terminal_status",
    "parse_event",
    "set_runtime",
]
