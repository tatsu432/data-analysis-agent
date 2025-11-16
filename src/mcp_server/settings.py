"""Settings for the MCP server."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the MCP server."""

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8082, validation_alias="PORT")
    transport: Literal["stdio", "http", "sse", "streamable-http"] = Field(
        default="http", validation_alias="TRANSPORT"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

