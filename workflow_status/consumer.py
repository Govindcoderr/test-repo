"""Background JetStream consumer that fans events into Redis.

Design points
-------------
* **One durable, every replica.** JetStream load-balances a durable across
  subscribers, so the same durable can safely be bound by every Tool-Engine
  pod — JetStream guarantees each event lands on exactly one of them. The
  pod writes to shared Redis, and whichever pod is waiting for that wdid
  reads it out. See package __init__.py for the full rationale.

* **At-least-once end-to-end.** A NATS message is only acked *after* the
  Redis write succeeds (or we proved it was a duplicate). If Redis hiccups
  mid-batch, we don't ack and NATS redelivers. ``jetstream_max_deliver``
  bounds retries so a permanently-bad event doesn't loop forever.

* **In-process subject filter.** The chosen durable
  (``communication-service-consumer``) was created with
  ``filter_subject=None`` — it serves every subject on the stream. We bind
  to the existing durable regardless, and drop anything whose subject
  doesn't match our configured full subject.

  WARNING: sharing a durable across services means events are *stolen*
  from the other binders (each message lands on only one subscriber). The
  loud warning in ``start()`` reminds operators of this at every boot.

* **Reconnect forever.** nats-py handles the TCP reconnect; we subscribe
  once per connection and re-subscribe on error with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.config import (
    WorkflowStatusSettings,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.models import (
    parse_event,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.store import (
    WorkflowEventStore,
)

logger = logging.getLogger("tool-engine.workflow-status.consumer")

_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 30.0


class WorkflowStatusConsumer:
    """Long-running JetStream pull consumer, run as a lifespan task."""

    def __init__(self, settings: WorkflowStatusSettings, store: WorkflowEventStore) -> None:
        self._settings = settings
        self._store = store
        self._task: asyncio.Task | None = None
        self._nc: Any = None  # nats.aio.client.Client
        self._js: Any = None
        self._stop = asyncio.Event()

        # Observability — read by /health. Never raise from the getter.
        self._connected = False
        self._subscribed = False
        self._last_error: str | None = None
        self._messages_received = 0
        self._messages_persisted = 0
        self._messages_dropped = 0

    # ─── Public lifecycle ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._settings.jetstream_enabled,
            "running": self.is_running(),
            "connected": self._connected,
            "subscribed": self._subscribed,
            "lastError": self._last_error,
            "messagesReceived": self._messages_received,
            "messagesPersisted": self._messages_persisted,
            "messagesDropped": self._messages_dropped,
            "subject": self._settings.full_subject,
            "stream": self._settings.jetstream_stream_name,
            "durable": self._settings.jetstream_durable_name,
        }

    async def start(self) -> None:
        if self._task is not None:
            return
        if not self._settings.is_configured():
            raise RuntimeError(
                "WorkflowStatusConsumer: subsystem is not fully configured — "
                "NATS_URL, JETSTREAM_STREAM_NAME, JETSTREAM_SUBJECT and "
                "JETSTREAM_DURABLE_NAME are all required."
            )

        # Loud, operator-facing warning: shared durables cause event stealing.
        # We can't detect "another service is using this" at runtime, so just
        # tell every operator every start.
        if "tool-engine" not in self._settings.jetstream_durable_name.lower():
            logger.warning(
                "JETSTREAM_DURABLE_NAME=%r does not contain 'tool-engine'. "
                "If this durable is shared with another service, events will be "
                "split between them (each message goes to exactly one subscriber). "
                "Set JETSTREAM_DURABLE_NAME to a Tool-Engine-specific value to avoid this.",
                self._settings.jetstream_durable_name,
            )

        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="workflow-status-consumer")
        logger.info(
            "WorkflowStatusConsumer started: stream=%s subject=%s durable=%s",
            self._settings.jetstream_stream_name,
            self._settings.full_subject,
            self._settings.jetstream_durable_name,
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            await self._close_nc()

    # ─── Internal: connect + fetch loop ──────────────────────────────────

    async def _run(self) -> None:
        backoff = _BACKOFF_INITIAL
        while not self._stop.is_set():
            try:
                await self._connect()
                sub = await self._subscribe()
                self._subscribed = True
                backoff = _BACKOFF_INITIAL  # reset after a successful setup

                await self._fetch_loop(sub)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — loop must survive anything
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("JetStream consumer error — reconnecting in %.1fs", backoff)
                self._subscribed = False
                await self._close_nc()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(_BACKOFF_MAX, backoff * 2)

    async def _connect(self) -> None:
        import nats  # imported lazily so a disabled subsystem has zero cost

        servers = [s.strip() for s in self._settings.nats_url.split(",") if s.strip()]
        if not servers:
            raise RuntimeError("NATS_URL is empty")

        async def _on_disconnect() -> None:
            self._connected = False
            logger.warning("NATS disconnected")

        async def _on_reconnect() -> None:
            self._connected = True
            logger.info("NATS reconnected to %s", self._nc.connected_url.netloc if self._nc else "?")

        async def _on_closed() -> None:
            self._connected = False
            logger.info("NATS connection closed")

        async def _on_error(exc: Exception) -> None:
            self._last_error = f"{type(exc).__name__}: {exc}"
            # Include the type: TimeoutError / gaierror have an empty str(),
            # which used to log just "NATS error: " with nothing after it.
            logger.warning(
                "NATS error (%s): %s %s",
                self._settings.nats_url, type(exc).__name__, exc,
            )

        self._nc = await nats.connect(
            servers=servers,
            name="tool-engine-workflow-status",
            connect_timeout=self._settings.jetstream_connect_timeout_seconds,
            max_reconnect_attempts=-1,  # retry forever
            reconnect_time_wait=2,
            disconnected_cb=_on_disconnect,
            reconnected_cb=_on_reconnect,
            closed_cb=_on_closed,
            error_cb=_on_error,
        )
        self._connected = True
        self._js = self._nc.jetstream()
        logger.info("NATS connected to %s", servers)

    async def _resolve_stream_name(self) -> str:
        """``JETSTREAM_STREAM_NAME`` may be either a stream name or a subject
        pattern — resolve to the real stream name once per connect.
        """
        value = self._settings.jetstream_stream_name.strip()
        if not self._settings.stream_is_subject_pattern:
            return value
        resolved = await self._js.find_stream_name_by_subject(value)
        logger.info("Resolved stream for subject %r → %r", value, resolved)
        return resolved

    async def _subscribe(self) -> Any:
        """Bind to the durable. We prefer bind (no filter rewrite) so we
        don't accidentally mutate a durable another service relies on. If
        the durable doesn't exist we create one with our filter.
        """
        from nats.js.api import ConsumerConfig, DeliverPolicy
        from nats.js.errors import NotFoundError

        stream = await self._resolve_stream_name()
        durable = self._settings.jetstream_durable_name

        # Try to bind to an existing durable (no filter/config change).
        try:
            info = await self._js.consumer_info(stream, durable)
            logger.info(
                "Binding to existing durable %r (filter=%r, ack_wait=%ss, max_deliver=%s)",
                durable, info.config.filter_subject, info.config.ack_wait, info.config.max_deliver,
            )
            return await self._js.pull_subscribe_bind(durable=durable, stream=stream)
        except NotFoundError:
            logger.info(
                "Durable %r not found on stream %r — creating with filter=%s",
                durable, stream, self._settings.full_subject,
            )
            config = ConsumerConfig(
                durable_name=durable,
                filter_subject=self._settings.full_subject,
                ack_wait=self._settings.jetstream_ack_wait_seconds,
                max_deliver=self._settings.jetstream_max_deliver,
                deliver_policy=DeliverPolicy.NEW,
            )
            return await self._js.pull_subscribe(
                subject=self._settings.full_subject,
                durable=durable,
                stream=stream,
                config=config,
            )

    async def _fetch_loop(self, sub: Any) -> None:
        batch = max(1, int(self._settings.jetstream_fetch_batch))
        timeout = max(0.5, float(self._settings.jetstream_fetch_timeout_seconds))
        full_subject = self._settings.full_subject
        # Allow listening to just a prefix if the suffix is empty (not our
        # default, but keep it correct).
        subject_prefix = full_subject.rstrip(">").rstrip(".")

        while not self._stop.is_set():
            try:
                msgs = await sub.fetch(batch, timeout=timeout)
            except asyncio.TimeoutError:
                continue  # empty window — totally normal
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Any other error (connection dropped mid-fetch, server
                # closed the pull) bubbles up to _run() for resubscribe.
                self._last_error = f"fetch: {type(exc).__name__}: {exc}"
                raise

            for msg in msgs:
                await self._handle_message(msg, subject_prefix)

    async def _handle_message(self, msg: Any, subject_prefix: str) -> None:
        self._messages_received += 1

        # In-process subject filter. Needed because the shared durable has
        # filter_subject=None and delivers every subject on the stream.
        if subject_prefix and not msg.subject.startswith(subject_prefix):
            self._messages_dropped += 1
            await _safe_ack(msg)
            return

        # Decode payload — invalid JSON is garbage, don't let it redeliver.
        try:
            raw = json.loads(msg.data.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._messages_dropped += 1
            logger.warning("Dropping non-JSON message on %s", msg.subject)
            await _safe_ack(msg)
            return

        parsed = parse_event(raw)
        if parsed is None:
            self._messages_dropped += 1
            logger.warning(
                "Dropping event with no workflow_definition_id on %s", msg.subject
            )
            await _safe_ack(msg)
            return

        wdid = parsed["workflow_definition_id"]
        try:
            written = await self._store.append_event(wdid, parsed)
        except Exception as exc:  # noqa: BLE001 — don't ack; let it redeliver
            logger.exception(
                "Redis write failed for wdid=%s — leaving message unacked for redelivery: %s",
                wdid, exc,
            )
            return  # do NOT ack

        if written:
            self._messages_persisted += 1
            node = parsed.get("node") or {}
            print(
                f"[JETSTREAM] 2. EVENT    rediskey={wdid}  "
                f"workflow={parsed.get('workflow_status')}  "
                f"node={node.get('sequence')}:{node.get('status')}  -> saved to Redis",
                flush=True,
            )
        else:
            # Duplicate — safe to ack; the earlier copy is already stored.
            self._messages_dropped += 1

        await _safe_ack(msg)

    async def _close_nc(self) -> None:
        self._connected = False
        self._subscribed = False
        if self._nc is None:
            return
        try:
            await self._nc.drain()
        except Exception as exc:  # noqa: BLE001 — best-effort close
            logger.debug("NATS drain failed (ignored): %s", exc)
        self._nc = None
        self._js = None


async def _safe_ack(msg: Any) -> None:
    try:
        await msg.ack()
    except Exception as exc:  # noqa: BLE001 — ack errors shouldn't crash loop
        logger.debug("ack failed (will be redelivered): %s", exc)
