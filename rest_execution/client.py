"""REST execution transport: POST a finished DAG to the backend workflow-
start API.

Each call is a single stateless HTTP POST — a client is created per call,
matching how the DAG itself is unique per call (fresh node/edge ids every
time, see ``AgentRuntime/dag/dag_builder.py``).

Error taxonomy:
  ``RestExecutionConfigError`` — REST not configured (no TOOL_API_BASE_URL).
  ``RestConnectionError``      — transport fault (timeout, DNS, refused).
  ``RestToolError``            — backend returned non-2xx.
"""

from __future__ import annotations

from typing import Any

import httpx

from ai_os.app.infrastructure.engines.tool_engine.infrastructure.rest_execution.config import (
    RestExecutionSettings,
)


class RestExecutionConfigError(RuntimeError):
    """REST execution was never configured, or the config is unusable."""


class RestConnectionError(RuntimeError):
    """The backend could not be reached, or the request/transport itself failed."""


class RestToolError(RuntimeError):
    """The backend received the DAG and reported a failure (non-2xx)."""


async def start_workflow(
    settings: RestExecutionSettings,
    dag: dict[str, Any],
) -> dict[str, Any]:
    """POST ``dag`` to the backend's workflow-start endpoint.

    Returns the backend's response body as a dict on 2xx.
    Raises ``RestExecutionConfigError`` when no ``base_url`` is configured,
    ``RestConnectionError`` on transport failures, and ``RestToolError`` on
    non-2xx responses.
    """
    if not settings.is_configured():
        raise RestExecutionConfigError(
            "REST execution has no base_url configured (TOOL_API_BASE_URL)"
        )

    url = f"{settings.base_url.rstrip('/')}/{settings.start_path.strip('/')}"

    # No Authorization header: the backend currently runs without auth.
    # If auth is reintroduced, add:
    #   headers["Authorization"] = f"Bearer {settings.auth_token}"
    headers = {"Content-Type": "application/json"}

    # The DAG from AgentRuntime already carries the real user_id and
    # workSpace_id of the caller — nothing is injected or defaulted here.
    body: dict[str, Any] = dict(dag)

    try:
        async with httpx.AsyncClient(timeout=settings.timeout_seconds) as http_client:
            response = await http_client.post(url, json=body, headers=headers)
            print(f"[REST Execution]: {response.text}")
    except httpx.TimeoutException as exc:
        raise RestConnectionError(
            f"REST execution backend timed out after {settings.timeout_seconds}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise RestConnectionError(
            f"REST execution backend unreachable or failed: {exc}"
        ) from exc

    payload = _safe_payload(response)
    if 200 <= response.status_code < 300:
        return payload if isinstance(payload, dict) else {"result": payload}

    raise RestToolError(
        f"REST execution backend returned {response.status_code}: {payload!r}"
    )


def _safe_payload(response: httpx.Response) -> Any:
    """Extract the response body — never leaks auth header."""
    try:
        return response.json()
    except (ValueError, httpx.DecodingError):
        return response.text
