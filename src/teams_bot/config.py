"""Configuration settings for the Teams bot."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()


class TeamsBotSettings(BaseSettings):
    """Settings for the Microsoft Teams bot."""

    model_config = SettingsConfigDict(
        env_file=str(env_file) if env_file.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    microsoft_app_id: str = Field(
        default="",
        validation_alias="MICROSOFT_APP_ID",
        description="Azure App Registration / Bot Application (client) ID. Leave empty for local development with emulator.",
    )

    microsoft_app_password: str = Field(
        default="",
        validation_alias="MICROSOFT_APP_PASSWORD",
        description="Azure App Registration client secret (bot password). Leave empty for local development with emulator.",
    )

    langgraph_server_url: str = Field(
        default="http://localhost:2024",
        validation_alias="LANGGRAPH_SERVER_URL",
        description="LangGraph server URL",
    )

    langgraph_assistant_id: str = Field(
        ...,
        validation_alias="LANGGRAPH_ASSISTANT_ID",
        description="LangGraph assistant ID",
    )

    langgraph_graph_name: str = Field(
        default="data_analysis_agent",
        validation_alias="LANGGRAPH_GRAPH_NAME",
        description="LangGraph graph name",
    )

    port: int = Field(
        default=3978,
        validation_alias="TEAMS_BOT_PORT",
        description="Port for the Teams bot HTTP server",
    )


@lru_cache
def get_settings() -> TeamsBotSettings:
    """Get cached settings instance."""
    return TeamsBotSettings()

