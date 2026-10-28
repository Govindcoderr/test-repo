"""Shared constants for the tool engine core.

Kept in a standalone module so both ``engine.py`` (top-level) and
``core/executor.py`` (lower-level) can import from here without
creating a circular dependency.

The value here is the in-code fallback only — the actual default is
read from settings (``config/settings.py`` → ``default_timeout_seconds``)
which reads from ``DEFAULT_TIMEOUT_SECONDS`` in the service's ``.env``.
"""

DEFAULT_TIMEOUT_SECONDS = 600
"""System-wide maximum tool execution timeout (seconds).

Overridden at runtime by ``settings.default_timeout_seconds`` which reads
``DEFAULT_TIMEOUT_SECONDS`` from ``.env``. This constant is only the
in-code fallback for tests or direct callers that bypass the settings layer.
"""
