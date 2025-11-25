"""Settings for the LangGraph server."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file from project root
# Find project root by looking for .env file or pyproject.toml
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback: try current directory
    load_dotenv()


class LLMProvider(StrEnum):
    """Supported LLM providers for the chat node."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    QWEN_OLLAMA = "qwen_ollama"


class BaseLLMModelSettings(BaseModel):
    llm_model_name: str = Field(
        default="gpt-5.1",
        description="LLM Model Name",
    )
    temperature: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Temperature",
    )


class ApiKeySettings(BaseModel):
    api_key: str = Field(
        description="API Key",
    )

    @property
    def llm_params(self) -> dict[str, Any]:
        return {"api_key": self.api_key}


class OpenAISettings(BaseLLMModelSettings, ApiKeySettings):
    llm_model_provider: Literal[LLMProvider.OPENAI] = LLMProvider.OPENAI


class AnthropicSettings(BaseLLMModelSettings, ApiKeySettings):
    llm_model_provider: Literal[LLMProvider.ANTHROPIC] = LLMProvider.ANTHROPIC


class LocalLLMSettings(BaseLLMModelSettings):
    """Settings for local LLM hosting (Ollama, OpenAI-compatible servers).

    Supports models like:
    - qwen3-coder:latest
    - gpt-oss:20b
    - Any other Ollama models
    """

    llm_model_provider: Literal[LLMProvider.LOCAL] = LLMProvider.LOCAL
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the local LLM server (e.g., Ollama: http://localhost:11434/v1)",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for local LLM server (usually not required)",
    )
    tool_choice: Literal["auto", "required", "none"] | str | None = Field(
        default=None,
        description="Tool choice behavior: 'auto' (model decides), 'required' (must use a tool), 'none' (no tools), or specific tool name. Defaults to 'required' for code generation.",
    )

    @property
    def llm_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "base_url": self.base_url,
            "model_provider": "openai",  # Use OpenAI-compatible API
            # For local LLMs, provide a dummy API key if none is set
            # Most local servers (like Ollama) accept any API key or ignore it
            "api_key": self.api_key or "ollama",
        }
        return params


class QwenOllamaSettings(BaseLLMModelSettings):
    """Settings for Qwen models via Ollama.

    This uses Ollama's OpenAI-compatible API, same as LocalLLMSettings.
    """

    llm_model_provider: Literal[LLMProvider.QWEN_OLLAMA] = LLMProvider.QWEN_OLLAMA
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the Ollama server",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key (dummy key will be used if not provided)",
    )
    tool_choice: Literal["auto", "required", "none"] | str | None = Field(
        default="auto",
        description="Tool choice behavior: 'auto' (model decides), 'required' (must use a tool), 'none' (no tools), or specific tool name",
    )

    @property
    def llm_params(self) -> dict[str, Any]:
        """Return parameters for Ollama's OpenAI-compatible API.

        Note: These params are used by initialize_llm to create the model instance.
        """
        return {
            "base_url": self.base_url,
            "model_provider": "openai",  # Use OpenAI-compatible API
            "api_key": self.api_key or "ollama",
        }


LLMSettings = Annotated[
    OpenAISettings | AnthropicSettings | LocalLLMSettings | QwenOllamaSettings,
    Field(discriminator="llm_model_provider"),
]
OptionalLLMSettings = Annotated[
    OpenAISettings | AnthropicSettings | LocalLLMSettings | QwenOllamaSettings | None,
    Field(default=None, discriminator="llm_model_provider"),
]


class Settings(BaseSettings):
    """Settings for the LangGraph server."""

    model_config = SettingsConfigDict(
        env_file=str(env_file) if env_file.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    mcp_server_url: str = Field(
        default="http://localhost:8082/mcp",
        validation_alias="MCP_SERVER_URL",
        description="Unified MCP server URL (includes all tools: analysis, knowledge, confluence)",
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

    # LLM settings
    chat_llm: LLMSettings = Field(alias="CHAT_NODE")
    coding_llm: OptionalLLMSettings = Field(default=None, alias="CODING_NODE")
    verifier_llm: OptionalLLMSettings = Field(default=None, alias="VERIFIER_NODE")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    # Log configuration for debugging
    import logging

    logger = logging.getLogger(__name__)

    # Validate LLM settings
    if settings.chat_llm is None:
        logger.error(
            "❌ CHAT_NODE configuration is missing! "
            "Please set LLM configuration in your .env file.\n"
            "Example for OpenAI:\n"
            "  CHAT_NODE__llm_model_provider=openai\n"
            "  CHAT_NODE__llm_model_name=gpt-5.1\n"
            "  CHAT_NODE__temperature=0.1\n"
            "  CHAT_NODE__api_key=your_api_key\n"
            "\n"
            "Example for Anthropic:\n"
            "  CHAT_NODE__llm_model_provider=anthropic\n"
            "  CHAT_NODE__llm_model_name=claude-3-5-sonnet-20241022\n"
            "  CHAT_NODE__temperature=0.1\n"
            "  CHAT_NODE__api_key=your_api_key\n"
            "\n"
            "Example for Local LLM (Ollama) - Qwen:\n"
            "  CHAT_NODE__llm_model_provider=local\n"
            "  CHAT_NODE__llm_model_name=qwen3-coder:latest\n"
            "  CHAT_NODE__temperature=0.1\n"
            "  CHAT_NODE__base_url=http://localhost:11434/v1\n"
            "  CHAT_NODE__tool_choice=required\n"
            "\n"
            "Example for Local LLM (Ollama) - GPT-OSS:\n"
            "  CHAT_NODE__llm_model_provider=local\n"
            "  CHAT_NODE__llm_model_name=gpt-oss:20b\n"
            "  CHAT_NODE__temperature=0.1\n"
            "  CHAT_NODE__base_url=http://localhost:11434/v1\n"
            "  CHAT_NODE__tool_choice=required"
        )
        raise ValueError(
            "CHAT_NODE configuration is required. "
            "Please set LLM configuration in your .env file using the CHAT_NODE__ prefix. "
            "See the error log above for examples."
        )

    logger.info(
        "✅ Settings loaded: Main LLM Model='%s' (Provider='%s', Temperature=%.2f)",
        settings.chat_llm.llm_model_name,
        settings.chat_llm.llm_model_provider,
        settings.chat_llm.temperature,
    )
    if settings.coding_llm:
        logger.info(
            "✅ Settings loaded: Coding LLM Model='%s' (Provider='%s', Temperature=%.2f)",
            settings.coding_llm.llm_model_name,
            settings.coding_llm.llm_model_provider,
            settings.coding_llm.temperature,
        )
    else:
        logger.info(
            "ℹ️  No separate coding LLM configured - main LLM will be used for code generation"
        )
    if settings.verifier_llm:
        logger.info(
            "✅ Settings loaded: Verifier LLM Model='%s' (Provider='%s', Temperature=%.2f)",
            settings.verifier_llm.llm_model_name,
            settings.verifier_llm.llm_model_provider,
            settings.verifier_llm.temperature,
        )
    else:
        logger.info(
            "ℹ️  No separate verifier LLM configured - main LLM (with adjusted temperature) will be used for verification"
        )
    return settings
