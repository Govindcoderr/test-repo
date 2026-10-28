"""The Tool Engine core.

Deliberately free of any ``ai_os`` import. This class knows nothing about the
kernel: it takes a payload dict, runs tools, and returns a result dict. That is
what makes it runnable and testable standalone, and it keeps the kernel coupling
confined to ``kernel_adapter.py``.

Not this class's job:

- Choosing which tools to run. The caller already decided.
- Calling another engine. If a tool needs an LLM, control returns to the caller.
- Holding session state. Nothing persists between ``run()`` calls.
- Retrying. The kernel owns retry policy; this reports outcomes once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ai_os.app.infrastructure.engines.tool_engine.catalog.loader import ToolCatalog
from ai_os.app.infrastructure.engines.tool_engine.catalog.schema_overlay import load_overlay
from ai_os.app.infrastructure.engines.tool_engine.core import aggregator, planner, references
from ai_os.app.infrastructure.engines.tool_engine.core.constants import DEFAULT_TIMEOUT_SECONDS
from ai_os.app.infrastructure.engines.tool_engine.core.circuit_breaker import (
    BreakerConfig,
    BreakerState,
    CircuitBreaker,
)
from ai_os.app.infrastructure.engines.tool_engine.core.executor import breaker_key
from ai_os.app.infrastructure.engines.tool_engine.core.resolver import resolve
from ai_os.app.infrastructure.engines.tool_engine.domain.contracts import (
    ResultSummary,
    ToolError,
    ToolExecutionPayload,
    ToolExecutionResult,
)
from ai_os.app.infrastructure.engines.tool_engine.domain.enums import AggregateStatus, ExecutionMode
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway import (
    manager as mcp_manager,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.config import (
    MCPServerConfig,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.mcp_gateway.manager import (
    DiscoveryReport,
)
from ai_os.app.infrastructure.engines.tool_engine.infrastructure.native.tools import (
    register_native_tools,
)
from ai_os.app.infrastructure.engines.tool_engine.manifest import EngineManifestData, build_manifest

if TYPE_CHECKING:
    from ai_os.app.infrastructure.engines.tool_engine.infrastructure.credentials.credential_manager import (
        CredentialManager,
    )

logger = logging.getLogger("tool-engine.engine")





class ToolEngine:
    """Executes batches of tool calls."""

    def __init__(
        self,
        catalog: ToolCatalog | None = None,
        servers_path: Path | None = None,
        breaker: CircuitBreaker | None = None,
        credential_manager: "CredentialManager | None" = None,
    ) -> None:
        register_native_tools()
        self._catalog = catalog or ToolCatalog.from_file()
        self._catalog.apply_schema_overlay(load_overlay())
        self._servers_path = servers_path

        # Build the circuit breaker from settings when not injected. This
        # lets CIRCUIT_BREAKER_FAILURE_THRESHOLD and
        # CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS in .env override the
        # hardcoded defaults in circuit_breaker.py without touching that file.
        if breaker is None:
            from ai_os.app.config.settings import get_settings as _get_settings
            _s = _get_settings()
            breaker = CircuitBreaker(
                default_config=BreakerConfig(
                    failure_threshold=_s.circuit_breaker_failure_threshold,
                    recovery_timeout_seconds=_s.circuit_breaker_recovery_timeout_seconds,
                )
            )
        self._mcp_configs: dict[str, MCPServerConfig] = {}
        self._disabled_servers: frozenset[str] = frozenset()
        self._breaker = breaker
        # Only wired when CREDENTIAL_FALLBACK_ENABLED is true (main.py).
        # None means payload tokens only (prod).
        self._credential_manager = credential_manager

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def catalog(self) -> ToolCatalog:
        return self._catalog

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def circuit_status(self) -> dict[str, dict[str, Any]]:
        return self._breaker.snapshot()

    def tool_health(self, tool_name: str) -> str:
        entry = resolve(self._catalog, tool_name)
        if entry is None:
            return "HEALTHY"
        state = self._breaker.state_of(breaker_key(entry.name))
        if state is BreakerState.OPEN:
            return "UNAVAILABLE"
        if state is BreakerState.HALF_OPEN:
            return "DEGRADED"
        return "HEALTHY"

    def manifest(self) -> EngineManifestData:
        return build_manifest(self._catalog)

    def available_tools(self) -> list[str]:
        return self._catalog.names()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def discover_mcp_tools(
        self,
        preloaded_entries: dict | None = None,
    ) -> DiscoveryReport:
        """Connect to configured MCP servers and register whatever they advertise.

        Args:
            preloaded_entries: Optional map of ``{server_name: [DiscoveredTool]}``
                for servers whose tools were already fetched during a Phase-1
                discovery restart.  Passed straight through to
                ``mcp_manager.discover()`` which skips the HTTP call for those
                servers, using the pre-collected schemas instead.
        """
        report = await mcp_manager.discover(
            self._servers_path,
            preloaded_entries=preloaded_entries,
        )
        self._catalog.register_dynamic(report.entries)
        self._catalog.apply_schema_overlay(load_overlay())
        self._mcp_configs = report.configs
        self._disabled_servers = frozenset(report.servers_disabled)
        self._last_report = report
        return report

    async def rediscover_failed(self) -> list[str]:
        """Retry discovery for servers that failed at startup.

        Returns the names of servers that recovered. Servers already working
        and servers that were skipped (disabled, missing URL) are not touched.
        """
        report = getattr(self, "_last_report", None)
        if report is None or not report.servers_failed:
            return []
        retry = await mcp_manager.rediscover(
            report.failed_configs, servers_path=self._servers_path
        )
        recovered: list[str] = []
        for name in retry.servers_ok:
            entries = [e for e in retry.entries if e.server == name]
            self._mcp_configs[name] = retry.configs[name]
            report.entries.extend(entries)
            report.servers_ok.append(name)
            report.configs[name] = retry.configs[name]
            report.sources[name] = retry.sources.get(name, "live")
            report.servers_failed.pop(name, None)
            report.failed_configs.pop(name, None)
            recovered.append(name)
        for name, reason in retry.servers_failed.items():
            report.servers_failed[name] = reason
        if recovered:
            # register_dynamic replaces all discovered entries, so pass the
            # full set (startup servers + recovered ones).
            self._catalog.register_dynamic(report.entries)
            self._catalog.apply_schema_overlay(load_overlay())
        return recovered

    def discovery_report(self) -> "DiscoveryReport | None":
        return getattr(self, "_last_report", None)

    async def initialize(
        self,
        preloaded_entries: dict | None = None,
    ) -> DiscoveryReport:
        """Startup hook — runs MCP discovery."""
        return await self.discover_mcp_tools(preloaded_entries=preloaded_entries)

    # ── REST path ─────────────────────────────────────────────────────────────
    #
    # The backend's /workflowInit POST only **queues** a workflow. The 2xx body
    # is an acknowledgement carrying ``rediskey`` and ``workflow_definition_id``
    # — not the real outcome. The real status (``running`` / ``completed`` /
    # ``failed`` / ``cancelled``) and the per-node output arrive asynchronously
    # on a JetStream subject and are fanned into Redis by
    # ``workflow_status.consumer``. This method:
    #
    #   1. POSTs the DAG (unchanged).
    #   2. Extracts ``workflow_definition_id`` from the ack. We correlate by
    #      this field ONLY — rediskey is passed through for traceability.
    #   3. Awaits the folded result from ``workflow_status.tracker.wait``.
    #   4. Returns a stable dict shape with ``workflowStatus`` (NOT ``status``;
    #      AgentRuntime's REST-shape detector treats a top-level ``status`` key
    #      as a batch result and would misroute it).
    #
    # Degraded mode: when the JetStream subsystem is disabled or the ack has
    # no workflow id we never claim ``completed`` on the back of a 2xx. We
    # return ``queued`` with a warning, or a clear ``failed`` with a code.

    async def run_rest(self, dag: dict[str, Any]) -> dict[str, Any]:
        from ai_os.app.infrastructure.engines.tool_engine.infrastructure.rest_execution import (
            client as rest_client,
        )
        from ai_os.app.infrastructure.engines.tool_engine.infrastructure.rest_execution.config import (
            get_settings as get_rest_settings,
        )
        from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status import (
            WorkflowStatus,
            extract_workflow_definition_id,
            get_runtime,
        )
        from ai_os.app.infrastructure.engines.tool_engine.infrastructure.workflow_status.models import (
            extract_rediskey,
        )

        settings = get_rest_settings()
        # Fire the POST. Any failure here is deterministic — map it straight to
        # a ``failed`` outcome without touching NATS at all.
        try:
            ack = await rest_client.start_workflow(settings, dag)
        except rest_client.RestExecutionConfigError as exc:
            return _rest_failure("REST_NOT_CONFIGURED", str(exc), retryable=False)
        except rest_client.RestConnectionError as exc:
            return _rest_failure("REST_CONNECTION_ERROR", str(exc), retryable=True)
        except rest_client.RestToolError as exc:
            return _rest_failure("REST_BACKEND_ERROR", str(exc), retryable=False)

        # Both the DAG we sent and the backend's ack may carry rediskey — the
        # DAG value is authoritative because we built it. Trace only.
        rediskey = (
            dag.get("rediskey")
            if isinstance(dag.get("rediskey"), str) and dag.get("rediskey")
            else extract_rediskey(ack)
        )

        wdid = extract_workflow_definition_id(ack)

        # Correlation key for JetStream events. Observed live: the backend's
        # ack carries a NUMERIC workflow_definition_id (e.g. 2839), but every
        # status event published on JetStream carries the DAG's ``rediskey``
        # (e.g. "REDIS-2A6EEDBCF8C5") in both ``workflow_definition_id`` and
        # ``ws_connection_id``. Waiting on the numeric id never matches a
        # single event and always ends in ``timeout``, so the tracker waits on
        # the rediskey. The numeric id is kept only for the result/trace.
        correlation_key = rediskey or wdid
        logger.info(
            "run_rest ack: workflow_definition_id=%s rediskey=%s correlating on %s",
            wdid, rediskey, correlation_key,
        )
        if not correlation_key:
            return {
                "workflowStatus": WorkflowStatus.FAILED.value,
                "code": "ACK_MISSING_WORKFLOW_ID",
                "error": (
                    "Neither the DAG nor the backend ack carries a rediskey or "
                    "workflow_definition_id — cannot correlate with JetStream events."
                ),
                "retryable": False,
                "acknowledgement": ack,
                "rediskey": rediskey,
                "output": None,
                "nodes": [],
            }

        runtime = get_runtime()
        if runtime is None or not runtime.consumer.is_running():
            # No consumer available: tell the caller the workflow was queued
            # (we won't fabricate a success) and surface a warning so the
            # mismatch is visible in logs.
            return {
                "workflowStatus": WorkflowStatus.QUEUED.value,
                "workflowDefinitionId": wdid,
                "rediskey": rediskey,
                "acknowledgement": ack,
                "output": None,
                "nodes": [],
                "warning": (
                    "JetStream workflow-status subsystem is unavailable; "
                    "returning queued acknowledgement without awaiting the "
                    "actual execution outcome."
                ),
            }

        print(
            f"[JETSTREAM] 1. WAITING  rediskey={correlation_key}  "
            f"timeout={runtime.result_timeout_seconds}s",
            flush=True,
        )

        # Record who is waiting (observability only).
        await runtime.store.register_pending(
            correlation_key,
            rediskey=rediskey,
        )

        outcome = await runtime.tracker.wait(
            correlation_key,
            timeout_seconds=runtime.result_timeout_seconds,
        )
        # Weave the acknowledgement and trace fields back in — the tracker
        # knows nothing about either. The tracker reports the correlation key
        # as ``workflowDefinitionId``; restore the backend's real numeric id.
        outcome["workflowDefinitionId"] = wdid or correlation_key
        outcome["acknowledgement"] = ack
        outcome["rediskey"] = rediskey
        _print_node_results(outcome)
        return outcome

    # ── Main execution path ───────────────────────────────────────────────────

    async def run(
    self,
    payload: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ToolExecutionResult:

        try:
            parsed = ToolExecutionPayload.model_validate(payload)
        except ValidationError as exc:
            result = _invalid_payload(exc)

            print("INVALID PAYLOAD RESULT:", type(result), result)

            return result

        conflict = _reference_mode_conflict(parsed)

        if conflict is not None:
            print("REFERENCE CONFLICT RESULT:", type(conflict), conflict)

            return conflict

        results, elapsed_ms = await planner.run_batch(
            calls=parsed.tools,
            catalog=self._catalog,
            granted_permissions=parsed.permissions,
            timeout_seconds=timeout_seconds,
            mode=parsed.execution_mode,
            on_failure=parsed.on_failure,
            mcp_configs=self._mcp_configs,
            breaker=self._breaker,
            user_id=parsed.user_id,
            credential_manager=self._credential_manager,
            disabled_servers=self._disabled_servers,
        )

        result = aggregator.build(
            results,
            parsed.execution_mode,
            elapsed_ms,
        )

        print("AGGREGATOR RESULT TYPE:", type(result))
        print("AGGREGATOR RESULT:", result)

        if not isinstance(result, ToolExecutionResult):
            raise TypeError(
                f"aggregator.build() returned "
                f"{type(result).__name__}, expected ToolExecutionResult"
            )

        return result

    async def run_as_dict(
    self,
    payload: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:

        result = await self.run(payload, timeout_seconds)

        print("RUN_AS_DICT RESULT TYPE:", type(result))

        if not isinstance(result, ToolExecutionResult):
            raise TypeError(
                f"Expected ToolExecutionResult, got {type(result).__name__}"
            )

        result_dict = result.model_dump(
            by_alias=True,
            mode="json",
        )

        print("RUN_AS_DICT OUTPUT TYPE:", type(result_dict))

        return result_dict


# ── Module helpers ────────────────────────────────────────────────────────────


def _print_node_results(outcome: dict[str, Any]) -> None:
    """Debug: print the real per-node results folded from JetStream events."""
    import json

    print("\n" + "=" * 120)
    print("TOOL ENGINE <- WORKFLOW NODE EXECUTION RESULT")
    print("=" * 120)
    print("workflowStatus :", outcome.get("workflowStatus"))
    print("rediskey       :", outcome.get("rediskey"))
    print("eventsSeen     :", outcome.get("events_seen"))
    print("elapsedSeconds :", outcome.get("elapsed_seconds"))
    for node in outcome.get("nodes") or []:
        print("-" * 120)
        print(
            f"node sequence={node.get('sequence')} run_count={node.get('run_count')} "
            f"nodeId={node.get('nodeId')} status={node.get('status')}"
        )
        print("started_at     :", node.get("started_at"))
        print("completed_at   :", node.get("completed_at"))
        print("output         :")
        print(json.dumps(node.get("output"), indent=2, ensure_ascii=False, default=str))



def _rest_failure(code: str, message: str, *, retryable: bool) -> dict[str, Any]:
    """Shape a REST-call failure the same way whether it comes from the HTTP
    client, a missing ack field, or a timeout. Keeps the kernel_adapter
    mapping simple."""
    return {
        "workflowStatus": "failed",
        "code": code,
        "error": message,
        "retryable": retryable,
        "output": None,
        "nodes": [],
    }

def _reference_mode_conflict(parsed: ToolExecutionPayload) -> ToolExecutionResult | None:
    if parsed.execution_mode is not ExecutionMode.PARALLEL:
        return None
    offenders = [
        (call.tool_call_id, ref)
        for call in parsed.tools
        for ref in references.find_references(call.input)
    ]
    if not offenders:
        return None
    detail = ", ".join(f"{call_id} -> {ref}" for call_id, ref in offenders)
    return _failure(
        ExecutionMode.PARALLEL,
        ToolError(
            code="REFERENCE_IN_PARALLEL",
            message=(
                "$ref needs an earlier result, which PARALLEL mode cannot provide "
                f"because every call starts at once. Use SEQUENTIAL. Found: {detail}"
            ),
        ),
    )


def _failure(mode: ExecutionMode, error: ToolError) -> ToolExecutionResult:
    return ToolExecutionResult(
        status=AggregateStatus.FAILURE,
        execution_mode=mode,
        summary=ResultSummary(),
        results=[],
        total_duration_ms=0.0,
        error=error,
    )


def _invalid_payload(exc: ValidationError) -> ToolExecutionResult:
    return _failure(
        ExecutionMode.SEQUENTIAL,
        ToolError(
            code="INVALID_PAYLOAD",
            message=f"payload does not match the tool.execute contract: {exc.error_count()} "
            f"problem(s); first: {exc.errors()[0].get('msg', 'unknown')}",
        ),
    )


def build_engine(catalog_path: Path | None = None) -> ToolEngine:
    """Convenience constructor used by the local harness and tests."""
    return ToolEngine(ToolCatalog.from_file(catalog_path) if catalog_path else None)
