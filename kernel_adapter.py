"""Kernel-facing adapter. The only module in this package that touches ``ai_os``.

Requires the kernel repo (``ASMA-AI-Agent-Service``) importable, so this module
does not import unless that is on the path. Nothing else in ``tool_engine``
imports it, which is what keeps the rest of the package runnable standalone.

Wiring, per the kernel's own docs:

    from tool_engine.kernel_adapter import ToolEngineAdapter

    plugin_loader.load_engine(ToolEngineAdapter())

``BaseEngine`` already implements ``initialize``, ``health_check`` and
``shutdown``, so only ``execute`` is overridden here.

When this moves into the kernel repo it belongs at
``ai_os/app/infrastructure/engines/Tool/``, alongside ``Supervisor/``.
"""

import logging
from typing import Any

from ai_os.app.domain.models.engine import EngineManifest
from ai_os.app.domain.models.kernel_request import KernelRequest
from ai_os.app.domain.models.kernel_response import KernelResponse, ResponseStatus
from ai_os.app.infrastructure.engines.base_engine import BaseEngine

from ai_os.app.infrastructure.engines.tool_engine import kernel_mapping
from ai_os.app.infrastructure.engines.tool_engine.core.redaction import redacted_json
from ai_os.app.infrastructure.engines.tool_engine.engine import ToolEngine
from ai_os.app.infrastructure.engines.tool_engine.manifest import CAPABILITY_TOOL_EXECUTE

logger = logging.getLogger("tool-engine-service.adapter")

# A payload with top-level ``source == "API"`` is a REST/API DAG dispatch —
# built by AgentRuntime's tool_payload.build_rest_payload(). Checked BEFORE
# the standard ToolExecutionPayload path so a DAG never accidentally gets
# parsed as a batch of tool calls. (A real batch carries ``source`` per call
# inside ``tools[]``, never at the top level.)
TOOL_SOURCE_API = "API"


class ToolEngineAdapter(BaseEngine):
    """Exposes the Tool Engine as an AI OS engine serving ``tool.execute``."""

    def __init__(self, engine: ToolEngine | None = None) -> None:
        self._engine = engine or ToolEngine()
        data = self._engine.manifest()
        super().__init__(
            EngineManifest(
                engineId=data.engine_id,
                name=data.name,
                version=data.version,
                description=data.description,
                capabilities=data.capabilities,
                resources=data.resources,
                metadata=data.metadata,
                priority=data.priority,
            )
        )

    async def initialize(self, preloaded_entries: dict | None = None) -> None:
        """Discover MCP tools, log the outcome, then refresh the manifest.

        Args:
            preloaded_entries: Optional map of ``{server_name: [DiscoveredTool]}``
                from a Phase-1 discovery restart.  Forwarded to the engine so
                restart-mode servers are not re-queried over HTTP (they are now
                in OAuth21 execution mode and would return 401).

        Both ``resources`` (tool names) AND ``metadata`` (toolSchemas) are
        refreshed after discovery so the kernel's engine registry reflects the
        complete, post-discovery tool set — including input schemas that the Agent
        Runtime Worker needs to call tools correctly on the first attempt.

        The kernel's ``LifecycleManager`` registers every manifest *before* calling
        ``initialize()``, so tools discovered here reach ``EngineRegistry`` (which
        holds this same manifest object) immediately after this method returns.
        """
        report = await self._engine.initialize(preloaded_entries=preloaded_entries)

        if report.servers_failed:
            for srv, reason in sorted(report.servers_failed.items()):
                logger.warning(
                    "MCP server %r tools unavailable — %s",
                    srv, reason,
                )

        # Refresh both resources AND metadata — two halves of the same answer.
        # A tool name in resources with no schema in metadata leaves the Worker
        # guessing argument names; refreshing only resources would reintroduce
        # that for every MCP tool.
        refreshed = self._engine.manifest()
        self._manifest.resources = refreshed.resources
        self._manifest.metadata = refreshed.metadata

        # Pre-warm the web_search browser so the FIRST search doesn't pay the
        # ~1-2s Chrome launch cost on top of its own work — that cold-start
        # penalty was enough to push borderline queries past the kernel's 30s
        # timeout. Best-effort and fire-and-forget: launching the browser must
        # not delay or break startup, and a deployment that never calls
        # web_search simply never uses it.
        try:
            import asyncio

            from ai_os.app.infrastructure.engines.tool_engine.infrastructure.native.tools.web_search import (
                browser_manager,
            )

            asyncio.create_task(browser_manager.get_browser())
        except Exception:  # noqa: BLE001 — pre-warm must not break startup
            logger.exception("web_search browser pre-warm failed to start")

        await super().initialize()

    def discovery_report(self):
        return self._engine.discovery_report()

    async def rediscover_failed(self) -> list[str]:
        """Retry failed MCP servers and refresh the manifest when any recover.

        The manifest object is the one the kernel registry holds, so updating
        ``resources`` and ``metadata`` in place exposes the new tools. Whether
        the kernel re-reads it without a restart depends on the kernel side.
        """
        recovered = await self._engine.rediscover_failed()
        if recovered:
            refreshed = self._engine.manifest()
            self._manifest.resources = refreshed.resources
            self._manifest.metadata = refreshed.metadata
        return recovered

    async def shutdown(self) -> None:
        """Close the shared web_search Playwright browser, then shut down normally.

        The web_search native tool keeps one headless Chrome process alive for the
        service's lifetime (see native/tools/web_search/browser_manager.py). Closing
        it here — the engine's single shutdown path, driven by main.py's lifespan —
        prevents a leaked/zombie browser process. Best-effort: a failure to close the
        browser must not stop the rest of shutdown.
        """
        try:
            from ai_os.app.infrastructure.engines.tool_engine.infrastructure.native.tools.web_search import (
                browser_manager,
            )

            await browser_manager.shutdown()
        except Exception:  # noqa: BLE001 — shutdown must not raise
            logger.exception("web_search browser shutdown failed")

        await super().shutdown()

    async def execute(self, request: KernelRequest) -> KernelResponse:
        """Run the requested tools and translate the outcome for the kernel."""

        if request.capability and request.capability != CAPABILITY_TOOL_EXECUTE:
            return self._failure(
                request,
                code="UNSUPPORTED_CAPABILITY",
                message=f"this engine serves {CAPABILITY_TOOL_EXECUTE!r}, "
                        f"got {request.capability!r}",
            )

        # Request trace. The payload carries toolConfig.tokens, so it is only
        # ever logged through the redactor, and only at DEBUG.
        logger.info(
            "tool.execute received requestId=%s executionId=%s correlationId=%s timeout=%s",
            request.request_id,
            request.execution_id,
            request.correlation_id,
            request.timeout,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("tool.execute payload=%s", redacted_json(request.payload))

        # ── REST / API branch ────────────────────────────────────────────────
        if str(request.payload.get("source") or "").upper() == TOOL_SOURCE_API:
            return await self._execute_rest(request)

        # ── NATIVE / MCP branch ──────────────────────────────────────────────
        from ai_os.app.config.settings import get_settings as _get_settings

        _settings = _get_settings()

        effective_timeout = max(
            request.timeout or 0,
            _settings.default_timeout_seconds,
        )

        result = await self._engine.run(
            request.payload,
            timeout_seconds=effective_timeout,
        )

        result_payload = result.model_dump(
            by_alias=True,
            mode="json",
        )

        logger.info(
            "tool.execute done requestId=%s status=%s total=%d durationMs=%s",
            request.request_id,
            result.status.value,
            result.summary.total,
            result.total_duration_ms,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("tool.execute result=%s", redacted_json(result_payload))

        status = kernel_mapping.response_status(result)

        return KernelResponse(
            requestId=request.request_id,
            executionId=request.execution_id,
            parentRequestId=request.parent_request_id,
            correlationId=request.correlation_id,
            status=ResponseStatus(status),
            result=result_payload,
            metrics=kernel_mapping.response_metrics(result),
            warnings=kernel_mapping.response_warnings(result),
            errors=kernel_mapping.response_errors(result),
            metadata={"sourceEngineId": self.manifest.engine_id},
        )

    async def _execute_rest(self, request: KernelRequest) -> KernelResponse:
        """Handle ``{"source": "API", "dag": {...}}`` payloads.

        Extracts the pre-built DAG and calls ``engine.run_rest()``, which
        POSTs it to the backend workflow-start API. Completely separate
        from the NATIVE/MCP path: no ToolExecutionPayload parsing, no
        executor, no breaker, no token needed.

        Never raises — ``run_rest`` normalises every outcome into
        ``{"success": bool, "output"/"error": ...}``.
        """
        dag = request.payload.get("dag")
        if not isinstance(dag, dict):
            return self._failure(
                request,
                code="INVALID_REST_PAYLOAD",
                message="source 'API' requires a 'dag' object in the payload",
            )

        outcome = await self._engine.run_rest(dag)
        response = self._build_rest_response(request, outcome)
        logger.info(
            "REST outcome requestId=%s kernelStatus=%s workflowStatus=%s code=%s "
            "error=%s eventsSeen=%s elapsed=%ss",
            request.request_id,
            response.status.value,
            outcome.get("workflowStatus"),
            outcome.get("code"),
            outcome.get("error"),
            outcome.get("events_seen"),
            outcome.get("elapsed_seconds"),
        )
        return response

    def _build_rest_response(
        self,
        request: KernelRequest,
        outcome: dict[str, Any],
    ) -> KernelResponse:
        """Translate ``run_rest`` outcome → KernelResponse.

        Status mapping (workflow_status → ResponseStatus):

            completed  → SUCCESS
            failed     → FAILURE   (output+nodes still carried in result for debugging)
            cancelled  → CANCELLED
            timeout    → TIMEOUT   (partial nodes carried through)
            queued     → PARTIAL   (JetStream subsystem unavailable; see warning)

        The result key is deliberately named ``workflowStatus`` rather than
        ``status``. AgentRuntime's REST-shape detector treats any result
        with a top-level ``status`` key as a ToolExecutionResult batch and
        tries to unpack it, which would mis-route this response.
        """
        workflow_status = str(outcome.get("workflowStatus") or "").lower()

        # The result body is identical across statuses — only ResponseStatus
        # and the errors/warnings arrays differ. Keeping one shape makes the
        # caller's job (AgentRuntime kernel_callback) trivial.
        result = {
            "workflowStatus": workflow_status or "unknown",
            "workflowDefinitionId": outcome.get("workflowDefinitionId"),
            "rediskey": outcome.get("rediskey"),
            "output": outcome.get("output"),
            "nodes": outcome.get("nodes") or [],
            "acknowledgement": outcome.get("acknowledgement"),
            "lastStatus": outcome.get("last_status"),
            "eventsSeen": outcome.get("events_seen"),
            "elapsedSeconds": outcome.get("elapsed_seconds"),
            "error": outcome.get("error"),
        }

        metadata = {"sourceEngineId": self.manifest.engine_id}
        warnings: list[str] = []
        if outcome.get("warning"):
            warnings.append(str(outcome["warning"]))

        if workflow_status == "completed":
            return KernelResponse(
                requestId=request.request_id,
                executionId=request.execution_id,
                parentRequestId=request.parent_request_id,
                correlationId=request.correlation_id,
                status=ResponseStatus.SUCCESS,
                result=result,
                warnings=warnings,
                metadata=metadata,
            )

        if workflow_status == "cancelled":
            return KernelResponse(
                requestId=request.request_id,
                executionId=request.execution_id,
                parentRequestId=request.parent_request_id,
                correlationId=request.correlation_id,
                status=ResponseStatus.CANCELLED,
                result=result,
                warnings=warnings,
                errors=_build_rest_errors(outcome, default_code="WORKFLOW_CANCELLED"),
                metadata=metadata,
            )

        if workflow_status == "timeout":
            return KernelResponse(
                requestId=request.request_id,
                executionId=request.execution_id,
                parentRequestId=request.parent_request_id,
                correlationId=request.correlation_id,
                status=ResponseStatus.TIMEOUT,
                result=result,
                warnings=warnings,
                errors=_build_rest_errors(
                    outcome,
                    default_code="WORKFLOW_TIMEOUT",
                    default_message=(
                        "Timed out waiting for workflow terminal status on JetStream"
                    ),
                    retryable=True,
                ),
                metadata=metadata,
            )

        if workflow_status == "queued":
            # Backend accepted the DAG but we can't observe the outcome
            # (JetStream disabled or unavailable). Report PARTIAL so the
            # caller distinguishes it from a real success.
            return KernelResponse(
                requestId=request.request_id,
                executionId=request.execution_id,
                parentRequestId=request.parent_request_id,
                correlationId=request.correlation_id,
                status=ResponseStatus.PARTIAL,
                result=result,
                warnings=warnings
                or [
                    "Workflow queued; actual execution status not observed. "
                    "Enable JetStream consumer to receive terminal status."
                ],
                metadata=metadata,
            )

        # Everything else — including "failed" and the sentinel "unknown"
        # — is a FAILURE. Still carry the result body so AgentRuntime can
        # log what happened per-node.
        return KernelResponse(
            requestId=request.request_id,
            executionId=request.execution_id,
            parentRequestId=request.parent_request_id,
            correlationId=request.correlation_id,
            status=ResponseStatus.FAILURE,
            result=result,
            warnings=warnings,
            errors=_build_rest_errors(outcome, default_code="REST_EXECUTION_FAILED"),
            metadata=metadata,
        )


def _build_rest_errors(
    outcome: dict[str, Any],
    *,
    default_code: str,
    default_message: str | None = None,
    retryable: bool | None = None,
) -> list[dict[str, Any]]:
    """Build the KernelResponse.errors array from an outcome dict."""
    code = str(outcome.get("code") or default_code)
    message = (
        str(outcome.get("error"))
        if outcome.get("error")
        else (default_message or "REST tool execution failed")
    )
    is_retryable = (
        bool(outcome.get("retryable"))
        if "retryable" in outcome
        else bool(retryable)
    )
    return [{"code": code, "message": message, "retryable": is_retryable}]
