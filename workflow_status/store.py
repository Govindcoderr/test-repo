"""Redis event store for workflow-status correlation.

Why Redis sits between the JetStream consumer and the REST waiter
----------------------------------------------------------------
JetStream delivers each message to exactly one subscriber per durable, which
means the pod that receives an event is almost never the pod holding the
HTTP request that triggered the workflow. We cannot route by workflow id at
the NATS layer without proliferating subjects/durables; using a shared
Redis instance that every replica already talks to is the pragmatic fix.

Keys (TTL = ``workflow_event_ttl_seconds``, default 900s)
---------------------------------------------------------
  wf:events:{wdid}            — Redis **stream** of parsed-event JSON blobs.
                                 run_rest reads from id ``0`` so it never
                                 misses events that landed before it registered.
  wf:seen:{wdid}:{sha1}       — SET NX dedup marker. Second write loses.
  wf:pending:{wdid}           — Hash with rediskey/request_id/execution_id,
                                 used only for observability — not for
                                 correlation.
  wf:result:{wdid}            — Final folded result (JSON). Lets any replica
                                 (or a retry) read the outcome without having
                                 to replay the stream.

All I/O is async via ``redis.asyncio``. Every writing method pipelines the
TTL update with the data write so a crash between the two cannot leave a
leaky key.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.models import (
    event_fingerprint,
)

logger = logging.getLogger("tool-engine.workflow-status.store")

# Key factories kept at module scope so other modules (consumer, tracker)
# cannot accidentally construct a mismatched key.
_K_EVENTS = "wf:events:{wdid}".format
_K_SEEN = "wf:seen:{wdid}:{sha}".format
_K_PENDING = "wf:pending:{wdid}".format
_K_RESULT = "wf:result:{wdid}".format


class WorkflowEventStore:
    """Thin wrapper around aioredis that owns the key layout."""

    def __init__(self, client: aioredis.Redis, ttl_seconds: int) -> None:
        self._r = client
        self._ttl = max(60, int(ttl_seconds))

    # ─── Pending registration (observability + leak detection) ────────────

    async def register_pending(
        self,
        wdid: str,
        *,
        rediskey: str | None,
        request_id: str | None,
        execution_id: str | None,
    ) -> None:
        """Record that a REST call is waiting on this workflow id.

        Not consulted during correlation — correlation is strictly by
        ``workflow_definition_id``. The hash exists so dashboards and
        post-mortems can tell which request a wdid belonged to.
        """
        key = _K_PENDING(wdid=wdid)
        try:
            async with self._r.pipeline(transaction=False) as pipe:
                pipe.hset(
                    key,
                    mapping={
                        "rediskey": rediskey or "",
                        "requestId": request_id or "",
                        "executionId": execution_id or "",
                    },
                )
                pipe.expire(key, self._ttl)
                await pipe.execute()
        except Exception as exc:  # noqa: BLE001 — best-effort registry
            logger.warning("register_pending(%s) failed: %s", wdid, exc)

    async def clear_pending(self, wdid: str) -> None:
        try:
            await self._r.delete(_K_PENDING(wdid=wdid))
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.debug("clear_pending(%s) failed: %s", wdid, exc)

    # ─── Event ingestion (consumer side) ──────────────────────────────────

    async def append_event(self, wdid: str, parsed_event: dict[str, Any]) -> bool:
        """Idempotently append a parsed event to the per-wdid stream.

        Returns False when the event is a duplicate (same fingerprint) and
        has been dropped. Returns True when the event was actually written —
        the consumer uses this to decide whether its NATS ack is "progress"
        or "redelivery already handled". Either way the NATS message should
        be acked: the dedup marker makes repeats a no-op.
        """
        fingerprint = event_fingerprint(parsed_event)
        seen_key = _K_SEEN(wdid=wdid, sha=fingerprint)

        try:
            first = await self._r.set(seen_key, "1", nx=True, ex=self._ttl)
        except Exception as exc:  # noqa: BLE001
            # If the dedup check itself fails, prefer to drop (don't risk
            # duplicate state). Return False so caller doesn't double-count.
            logger.warning("dedup SET NX failed for %s: %s", wdid, exc)
            return False
        if not first:
            return False

        payload = json.dumps(parsed_event, default=str)
        stream_key = _K_EVENTS(wdid=wdid)
        try:
            async with self._r.pipeline(transaction=False) as pipe:
                pipe.xadd(stream_key, {"data": payload})
                pipe.expire(stream_key, self._ttl)
                await pipe.execute()
        except Exception:
            # Roll back the dedup marker so a retry can succeed. This keeps
            # at-least-once end-to-end: the NATS message won't be acked by
            # the consumer, redelivery will try again.
            try:
                await self._r.delete(seen_key)
            except Exception:  # noqa: BLE001
                pass
            raise
        return True

    # ─── Event consumption (tracker side) ─────────────────────────────────

    async def read_events_blocking(
        self,
        wdid: str,
        *,
        last_id: str,
        block_ms: int,
        count: int = 50,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Blocking read from ``wf:events:{wdid}``.

        ``last_id`` is a Redis stream id — pass ``"0"`` on the first call to
        drain anything already buffered, then the id returned by the
        previous call to continue.

        Returns a list of ``(entry_id, parsed_event)`` tuples. Empty list on
        timeout; the tracker uses that to re-check its deadline and loop.
        """
        stream_key = _K_EVENTS(wdid=wdid)
        try:
            resp = await self._r.xread(
                streams={stream_key: last_id},
                count=count,
                block=max(0, int(block_ms)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("xread(%s) failed: %s", wdid, exc)
            # Back off before the tracker loops again. Without this a broken
            # Redis connection turns the wait loop into a hot retry that
            # floods the log and hammers the server.
            await asyncio.sleep(1.0)
            return []

        out: list[tuple[str, dict[str, Any]]] = []
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                raw = fields.get("data") if isinstance(fields, dict) else None
                if raw is None:
                    # redis-py returns bytes-or-str depending on
                    # decode_responses — tolerate both.
                    raw = fields.get(b"data") if isinstance(fields, dict) else None
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("corrupt event payload at %s — skipping", entry_id)
                    continue
                entry_id_str = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                out.append((entry_id_str, parsed))
        return out

    # ─── Final result cache ───────────────────────────────────────────────

    async def store_result(self, wdid: str, result: dict[str, Any]) -> None:
        try:
            await self._r.set(
                _K_RESULT(wdid=wdid),
                json.dumps(result, default=str),
                ex=self._ttl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("store_result(%s) failed: %s", wdid, exc)

    async def load_result(self, wdid: str) -> dict[str, Any] | None:
        try:
            raw = await self._r.get(_K_RESULT(wdid=wdid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_result(%s) failed: %s", wdid, exc)
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
