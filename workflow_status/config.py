"""Workflow-status subsystem settings.

Reads NATS + JetStream + Redis-buffer values straight from the process
environment (``.env`` via pydantic-settings). No prefix: the keys are shared
with the Agent-Runtime service, so we match them exactly:

    NATS_URL
    JETSTREAM_STREAM_NAME             (stream name OR subject pattern)
    JETSTREAM_SUBJECT                 (prefix; the suffix is appended)
    JETSTREAM_SUBJECT_SUFFIX          (default: a0.agent.execution.status.update)
    JETSTREAM_DURABLE_NAME
    JETSTREAM_RESULT_TIMEOUT_SECONDS
    JETSTREAM_ACK_WAIT_SECONDS
    JETSTREAM_MAX_DELIVER
    JETSTREAM_FETCH_BATCH
    JETSTREAM_FETCH_TIMEOUT_SECONDS
    JETSTREAM_CONNECT_TIMEOUT_SECONDS
    JETSTREAM_ENABLED

    WORKFLOW_EVENT_TTL_SECONDS
    WORKFLOW_STATUS_REDIS_URL         (optional; falls back to QUEUE_REDIS_URL)
    QUEUE_REDIS_URL                   (fallback)

The ``full_subject`` property composes the actual filter subject the consumer
subscribes to. We keep subject and suffix separate so operators can change
either half independently without a code change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_STREAM_IS_SUBJECT_MARKERS = (".", ">", "*")


class WorkflowStatusSettings(BaseSettings):
    """Deployment settings for the JetStream workflow-status subsystem."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Master switch. When false the consumer is not started and run_rest()
    # falls back to the queued-only degraded response.
    jetstream_enabled: bool = True

    # NATS connection
    nats_url: str = ""
    jetstream_connect_timeout_seconds: float = 10.0

    # Stream + subject
    jetstream_stream_name: str = ""
    """Either a stream NAME (e.g. ``dev-stream``) or a subject pattern
    (``dev.>``) — if the value contains a dot or a wildcard, the consumer
    resolves the real stream name via ``find_stream_name_by_subject``."""

    jetstream_subject: str = ""
    """Prefix shared by every workflow status event. Combined with the
    suffix to form the full filter subject."""

    jetstream_subject_suffix: str = "agent.execution.status.update"

    # Consumer
    jetstream_durable_name: str = ""
    jetstream_result_timeout_seconds: float = 240.0
    jetstream_ack_wait_seconds: float = 30.0
    jetstream_max_deliver: int = 5
    jetstream_fetch_batch: int = 50
    jetstream_fetch_timeout_seconds: float = 5.0

    # Redis buffer
    workflow_event_ttl_seconds: int = 900
    workflow_status_redis_url: str = ""
    queue_redis_url: str = ""

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def full_subject(self) -> str:
        """The subject the consumer filters on in-process."""
        prefix = self.jetstream_subject.strip().strip(".")
        suffix = self.jetstream_subject_suffix.strip().strip(".")
        if prefix and suffix:
            print(f"full_subject: {prefix}.{suffix}")
            return f"{prefix}.{suffix}"
        return prefix or suffix

    @property
    def stream_is_subject_pattern(self) -> bool:
        value = self.jetstream_stream_name.strip()
        return any(marker in value for marker in _STREAM_IS_SUBJECT_MARKERS)

    def effective_redis_url(self) -> str:
        """Pick a Redis URL for the event buffer.

        Priority: WORKFLOW_STATUS_REDIS_URL → QUEUE_REDIS_URL → localhost.
        The last fallback keeps local development working out of the box;
        production deployments always set one of the first two.
        """
        if self.workflow_status_redis_url:
            return self.workflow_status_redis_url
        if self.queue_redis_url:
            return self.queue_redis_url
        return "redis://localhost:6379/0"

    def is_configured(self) -> bool:
        """True when the minimum knobs needed to start the consumer are set."""
        return bool(
            self.jetstream_enabled
            and self.nats_url
            and self.jetstream_stream_name
            and self.full_subject
            and self.jetstream_durable_name
        )


@lru_cache
def get_settings() -> WorkflowStatusSettings:
    """Cached settings instance — .env is read once."""
    return WorkflowStatusSettings()
