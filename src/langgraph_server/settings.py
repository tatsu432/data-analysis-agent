"""Settings for the LangGraph server."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the LangGraph server."""

    data_analysis_mcp_server_url: str = Field(
        default="http://localhost:8082/mcp",
        validation_alias="DATA_ANALYSIS_MCP_SERVER_URL",
    )
    mcp_server_timeout: int = Field(default=30, validation_alias="MCP_SERVER_TIMEOUT")
    mcp_server_sse_read_timeout: int = Field(
        default=60, validation_alias="MCP_SERVER_SSE_READ_TIMEOUT"
    )
    mcp_server_terminate_on_close: bool = Field(
        default=True, validation_alias="MCP_SERVER_TERMINATE_ON_CLOSE"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
