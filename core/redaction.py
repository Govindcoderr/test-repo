"""Remove tokens and secrets from anything that is about to be logged.

Every payload, Tool_Call, toolConfig or header set must go through
``redact()`` before it reaches a log line. The function works on a copy, so the
object that is forwarded to the MCP server keeps its real token.

Two layers:

- Key-based: the value of any key in ``SENSITIVE_KEYS`` (case-insensitive) is
  replaced with ``[REDACTED]`` at every nesting depth, whatever its type.
- Text-based: ``Bearer <token>`` inside any string keeps the ``Bearer `` prefix
  and loses the token.
"""

from __future__ import annotations

import json
import re
from typing import Any

REDACTED = "[REDACTED]"
REDACTION_FAILED = "[REDACTION_FAILED]"

SENSITIVE_KEYS: frozenset[str] = frozenset(
    k.lower()
    for k in (
        "accessToken",
        "access_token",
        "refreshToken",
        "refresh_token",
        "clientSecret",
        "client_secret",
        "api_key",
        "apiKey",
        "token",
        "id_token",
        "idToken",
        "Authorization",
        "password",
    )
)

_BEARER = re.compile(r"(Bearer\s+)[^\s\"',;]+", re.IGNORECASE)


def redact_text(text: str) -> str:
    """Replace ``Bearer <token>`` occurrences inside free text."""
    return _BEARER.sub(lambda m: m.group(1) + REDACTED, text)


def redact(value: Any) -> Any:
    """Return a redacted deep copy of ``value``. Never mutates the input.

    Cyclic or otherwise unprocessable values come back as ``REDACTION_FAILED``
    instead of raising, so logging can never break a tool call.
    """
    try:
        return _redact(value, set())
    except Exception:  # noqa: BLE001 - logging must never break execution
        return REDACTION_FAILED


def _redact(value: Any, seen: set[int]) -> Any:
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return REDACTION_FAILED
        seen = seen | {marker}
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in SENSITIVE_KEYS:
                out[key] = REDACTED
            else:
                out[key] = _redact(item, seen)
        return out
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in seen:
            return REDACTION_FAILED
        seen = seen | {marker}
        items = [_redact(item, seen) for item in value]
        return items if isinstance(value, list) else tuple(items)
    if isinstance(value, str):
        return redact_text(value)
    return value


def redacted_json(value: Any) -> str:
    """JSON dump of the redacted value, for one-line debug logs."""
    try:
        return json.dumps(redact(value), ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return REDACTION_FAILED
