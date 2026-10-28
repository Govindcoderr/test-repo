"""Process-wide lifecycle for the workflow-status subsystem.

Keeps the Tool Engine *the engine* free of singleton bookkeeping. The
engine looks the runtime up through ``get_runtime()`` when it needs the
tracker; everything about *when* the Redis client and consumer come up
lives here, driven by ``main.py``'s FastAPI lifespan.

Two levels of enablement:

  * ``settings.jetstream_enabled`` false  → no runtime at all. ``run_rest``
    sees ``get_runtime() is None`` and degrades to the queued response.
  * enabled but ``is_configured()`` false → explicit failure at startup
    rather than silent degradation; a misconfigured prod env should not
    pretend to work.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.config import (
    WorkflowStatusSettings,
    get_settings,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.consumer import (
    WorkflowStatusConsumer,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.store import (
    WorkflowEventStore,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.tracker import (
    WorkflowTracker,
)

logger = logging.getLogger("tool-engine.workflow-status.runtime")

_runtime: "WorkflowStatusRuntime | None" = None


class WorkflowStatusRuntime:
    """Owns the Redis client, the consumer task and the tracker instance."""

    def __init__(
        self,
        settings: WorkflowStatusSettings,
        redis_client: aioredis.Redis,
        store: WorkflowEventStore,
        consumer: WorkflowStatusConsumer,
        tracker: WorkflowTracker,
    ) -> None:
        self.settings = settings
        self.redis = redis_client
        self.store = store
        self.consumer = consumer
        self.tracker = tracker

    @property
    def result_timeout_seconds(self) -> float:
        return self.settings.jetstream_result_timeout_seconds

    async def start(self) -> None:
        await self.consumer.start()

    async def shutdown(self) -> None:
        try:
            await self.consumer.stop()
        finally:
            try:
                await self.redis.aclose()  # redis>=5 async close
            except AttributeError:
                # Older redis-py: fall back to the deprecated close()
                try:
                    await self.redis.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("redis close failed (ignored): %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("redis aclose failed (ignored): %s", exc)

    def status(self) -> dict:
        return self.consumer.status()
    


async def build_runtime(settings: WorkflowStatusSettings | None = None) -> "WorkflowStatusRuntime | None":
    """Build and return a runtime, or ``None`` when the subsystem is disabled.

    Does NOT start the consumer — callers (lifespan) call ``start()`` after
    the runtime is set as the process singleton so a race on ``get_runtime``
    cannot see a half-initialised state.
    """
    settings = settings or get_settings()
    if not settings.jetstream_enabled:
        logger.info("WorkflowStatus subsystem disabled (JETSTREAM_ENABLED=false)")
        return None
    if not settings.is_configured():
        raise RuntimeError(
            "WorkflowStatus subsystem is enabled but mis-configured. Required: "
            "NATS_URL, JETSTREAM_STREAM_NAME, JETSTREAM_SUBJECT, JETSTREAM_DURABLE_NAME."
        )

    redis_url = settings.effective_redis_url()
    client = aioredis.from_url(
        redis_url,
        decode_responses=True,
        # Must exceed the longest XREAD BLOCK (tracker._MAX_BLOCK_MS = 5s)
        # plus network latency to the cloud Redis, otherwise an idle
        # blocking read is reported as "Timeout reading from ...".
        socket_timeout=30.0,
        socket_connect_timeout=5.0,
        health_check_interval=30,
        # Keep the TCP connection alive and auto-retry when a read/write
        # times out. On Windows an idle cloud-Redis socket can be dropped
        # mid-wait, surfacing as "WinError 121: semaphore timeout"; these two
        # make the client reconnect and retry instead of failing the call.
        socket_keepalive=True,
        retry_on_timeout=True,
    )
    # Fail fast if Redis is wrong, instead of surfacing it on the first REST call.
    try:
        await client.ping()
    except Exception:
        try:
            await client.aclose()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        raise

    store = WorkflowEventStore(client, ttl_seconds=settings.workflow_event_ttl_seconds)
    tracker = WorkflowTracker(store)
    consumer = WorkflowStatusConsumer(settings, store)
    return WorkflowStatusRuntime(settings, client, store, consumer, tracker)



def set_runtime(runtime: "WorkflowStatusRuntime | None") -> None:
    """Install (or clear) the process-wide runtime. Called by main.py."""
    global _runtime
    _runtime = runtime


def get_runtime() -> "WorkflowStatusRuntime | None":
    """Return the active runtime, or ``None`` when disabled/unavailable.

    ``run_rest`` uses this to decide between the full wait-for-result path
    and the degraded ``queued`` response.
    """
    return _runtime
