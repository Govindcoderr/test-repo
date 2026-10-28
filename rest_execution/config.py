"""REST execution backend settings: base URL, auth token, timeout.

Loaded from ``.env`` via ``pydantic_settings`` with prefix ``TOOL_API_``.
No secret is ever hardcoded here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RestExecutionSettings(BaseSettings):
    """Deployment-level settings for the REST tool-execution transport."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TOOL_API_",
        extra="ignore",
    )

    base_url: str = ""
    """Base URL of the backend workflow-start API — ``TOOL_API_BASE_URL``.
    Empty means REST execution is unconfigured — every REST tool call then
    fails closed with a clear ``RestExecutionConfigError`` rather than a
    confusing connection error."""

    start_path: str = "/A0-service/api/v1/workflowInit/agent/start"
    """Path REST-execution DAGs are POSTed to — ``TOOL_API_START_PATH``."""

    auth_token: str = ""
    """Bearer token for the backend — ``TOOL_API_AUTH_TOKEN``. Never logged,
    never echoed into a tool result."""

    timeout_seconds: float = 60.0
    """Ceiling for the HTTP request — ``TOOL_API_TIMEOUT_SECONDS``."""

    def is_configured(self) -> bool:
        return bool(self.base_url)


@lru_cache
def get_settings() -> RestExecutionSettings:
    """Cached settings instance. Cached because ``.env`` is read from disk."""
    return RestExecutionSettings()
