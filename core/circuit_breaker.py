"""Circuit breaker: stop calling a tool that is already failing.

States
------

.. code-block:: text

    CLOSED --(failures reach threshold)--> OPEN
      ^                                     |
      |                          (recovery timeout elapses,
      |                           next call arrives)
      |                                     v
      +--(trial succeeds)---- HALF_OPEN ----+
                                  |  (trial fails)
                                  +--> OPEN

``CLOSED``  Normal service. Calls dispatch; failures are counted.
``OPEN``    Fail fast. Nothing dispatches — zero cost on every rejected call.
``HALF_OPEN`` One call is let through to see whether the tool recovered.

Only transient failures count toward the threshold. ``classify`` in
``core/errors.py`` marks a failure ``retryable`` when a repeat attempt could
plausibly succeed (transport fault, timeout). A non-retryable failure (bad
arguments, permission denied) does not move the breaker.

The transition out of OPEN is triggered by a call, not by a timer — a breaker
nobody is calling has nothing to prove and a timer per key would be costly.

In-memory, per process. State is lost on restart; a gateway with multiple
instances has one breaker per instance rather than a shared view.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any

__all__ = [
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RECOVERY_TIMEOUT_SECONDS",
    "BreakerConfig",
    "BreakerState",
    "CircuitBreaker",
]

DEFAULT_FAILURE_THRESHOLD = 5
"""Consecutive transient failures before a breaker opens."""

DEFAULT_RECOVERY_TIMEOUT_SECONDS = 30.0
"""How long a breaker stays OPEN before a trial call is allowed.
Override via TOOL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS in .env —
see tool_engine/config/settings.py and ToolEngine.__init__ in engine.py."""


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    """Threshold and timeout for one key, or the default for all of them."""

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    recovery_timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be positive")


@dataclass
class _KeyState:
    """Mutable breaker state for a single key."""

    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None
    trial_started_at: float | None = None

    # Counters for metrics / observability
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0
    times_opened: int = 0
    transitions: list[str] = field(default_factory=list)


_MAX_TRANSITIONS = 20


class CircuitBreaker:
    """Per-key circuit breaker for tool dispatch.

    A *key* is whatever failing unit should be isolated — typically a tool name.
    One key per downstream dependency so one broken integration cannot take out
    the tools that still work.

    Not a singleton and not module-level state: the engine owns one instance, a
    test owns its own. Shared mutable state that tests cannot isolate is how a
    suite starts passing and failing depending on order.
    """

    def __init__(
        self,
        default_config: BreakerConfig | None = None,
        overrides: dict[str, BreakerConfig] | None = None,
    ) -> None:
        self._default = default_config or BreakerConfig()
        self._overrides = dict(overrides or {})
        self._keys: dict[str, _KeyState] = {}
        self._lock = asyncio.Lock()

    # ── configuration ───────────────────────────────────────────────────

    def configure(self, key: str, config: BreakerConfig) -> None:
        """Set a per-key threshold and timeout, overriding the default."""
        self._overrides[key] = config

    def config_for(self, key: str) -> BreakerConfig:
        return self._overrides.get(key, self._default)

    # ── the gate ────────────────────────────────────────────────────────

    async def allow(self, key: str) -> bool:
        """Whether a call for ``key`` may be dispatched right now.

        Called *before* dispatch. ``False`` here is the difference between a
        refusal that costs nothing and a timeout that costs the full budget.

        Transitions OPEN to HALF_OPEN as a side effect when the recovery timeout
        has elapsed, because this is the only moment anything looks at the clock.
        """
        async with self._lock:
            entry = self._keys.setdefault(key, _KeyState())
            config = self.config_for(key)
            now = monotonic()

            if entry.state is BreakerState.CLOSED:
                return True

            if entry.state is BreakerState.OPEN:
                if entry.opened_at is not None and (
                    now - entry.opened_at < config.recovery_timeout_seconds
                ):
                    entry.total_rejected += 1
                    return False
                # Timeout elapsed: this call becomes the trial.
                self._transition(entry, BreakerState.HALF_OPEN)
                entry.trial_started_at = now
                return True

            # HALF_OPEN: one trial at a time.
            if entry.trial_started_at is None:
                entry.trial_started_at = now
                return True

            if now - entry.trial_started_at >= config.recovery_timeout_seconds:
                # The previous trial never reported back (likely cancelled).
                # Treat it as abandoned so the breaker doesn't wedge HALF_OPEN.
                entry.trial_started_at = now
                return True

            entry.total_rejected += 1
            return False

    # ── outcome reporting ───────────────────────────────────────────────

    async def record_success(self, key: str) -> None:
        """Report that a dispatched call for ``key`` succeeded."""
        async with self._lock:
            entry = self._keys.setdefault(key, _KeyState())
            entry.total_successes += 1
            entry.consecutive_failures = 0
            entry.trial_started_at = None

            if entry.state is not BreakerState.CLOSED:
                self._transition(entry, BreakerState.CLOSED)
                entry.opened_at = None

    async def record_failure(self, key: str, *, retryable: bool) -> None:
        """Report that a dispatched call for ``key`` failed.

        ``retryable`` comes from ``core/errors.classify`` and decides whether this
        failure counts toward opening the breaker. A non-retryable failure is still
        recorded for metrics but does not move the breaker.
        """
        async with self._lock:
            entry = self._keys.setdefault(key, _KeyState())
            entry.total_failures += 1

            if not retryable:
                entry.trial_started_at = None
                return

            entry.consecutive_failures += 1
            entry.trial_started_at = None

            if entry.state is BreakerState.HALF_OPEN:
                self._open(entry)
                return

            if entry.consecutive_failures >= self.config_for(key).failure_threshold:
                self._open(entry)

    # ── observation ─────────────────────────────────────────────────────

    def state_of(self, key: str) -> BreakerState:
        """Current state without creating an entry for an unseen key."""
        entry = self._keys.get(key)
        return entry.state if entry is not None else BreakerState.CLOSED

    def is_open(self, key: str) -> bool:
        return self.state_of(key) is BreakerState.OPEN

    def get_state(self, key: str) -> dict[str, Any]:
        """Snapshot for one key. Safe to log; contains no request content."""
        entry = self._keys.get(key)
        config = self.config_for(key)
        if entry is None:
            return {
                "key": key,
                "state": str(BreakerState.CLOSED),
                "consecutiveFailures": 0,
                "failureThreshold": config.failure_threshold,
                "recoveryTimeoutSeconds": config.recovery_timeout_seconds,
                "totalFailures": 0,
                "totalSuccesses": 0,
                "totalRejected": 0,
                "timesOpened": 0,
                "secondsInOpen": None,
                "transitions": [],
            }

        return {
            "key": key,
            "state": str(entry.state),
            "consecutiveFailures": entry.consecutive_failures,
            "failureThreshold": config.failure_threshold,
            "recoveryTimeoutSeconds": config.recovery_timeout_seconds,
            "totalFailures": entry.total_failures,
            "totalSuccesses": entry.total_successes,
            "totalRejected": entry.total_rejected,
            "timesOpened": entry.times_opened,
            "secondsInOpen": (
                round(monotonic() - entry.opened_at, 3) if entry.opened_at is not None else None
            ),
            "transitions": list(entry.transitions),
        }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Snapshot of every key seen so far, for a metrics or status endpoint."""
        return {key: self.get_state(key) for key in self._keys}

    # ── internal helpers ────────────────────────────────────────────────

    def _open(self, entry: _KeyState) -> None:
        entry.opened_at = monotonic()
        entry.times_opened += 1
        self._transition(entry, BreakerState.OPEN)

    def _transition(self, entry: _KeyState, new_state: BreakerState) -> None:
        label = f"{entry.state}->{new_state}"
        entry.state = new_state
        entry.transitions.append(label)
        if len(entry.transitions) > _MAX_TRANSITIONS:
            entry.transitions = entry.transitions[-_MAX_TRANSITIONS:]
