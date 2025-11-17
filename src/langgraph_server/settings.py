"""Settings for the LangGraph server."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env file from project root
# Find project root by looking for .env file or pyproject.toml
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback: try current directory
    load_dotenv()


class Settings(BaseSettings):
    """Settings for the LangGraph server."""

    data_analysis_mcp_server_url: str = Field(
        default="http://localhost:8082/mcp",
        validation_alias="DATA_ANALYSIS_MCP_SERVER_URL",
    )
    confluence_mcp_server_url: str | None = Field(
        default=None,
        validation_alias="CONFLUENCE_MCP_SERVER_URL",
    )
    confluence_space_key_analytics: str = Field(
        default="ANALYTICS",
        validation_alias="CONFLUENCE_SPACE_KEY_ANALYTICS",
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
    settings = Settings()
    # Log Confluence configuration for debugging
    import logging

    logger = logging.getLogger(__name__)
    if settings.confluence_mcp_server_url:
        logger.info(
            "✅ Settings loaded: CONFLUENCE_MCP_SERVER_URL=%s",
            settings.confluence_mcp_server_url,
        )
    else:
        logger.warning(
            "⚠️  Settings: CONFLUENCE_MCP_SERVER_URL is None (not set in environment)"
        )
    return settings
