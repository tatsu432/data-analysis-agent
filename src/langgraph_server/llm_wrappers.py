"""Custom LLM wrapper classes for better LangSmith visibility.

This module provides wrapper classes for local LLM models to ensure
they show up correctly in LangSmith traces with custom type names.
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class QwenOllama(ChatOpenAI):
    """Wrapper for Qwen models via Ollama that shows up as 'qwen_ollama' in LangSmith.

    This class extends ChatOpenAI to work with Ollama's OpenAI-compatible API
    while ensuring LangSmith traces show the correct model type name.
    """

    @property
    def _llm_type(self) -> str:
        """Return the LLM type name for LangSmith tracing."""
        return "ChatQwenOllama"

    @classmethod
    def create(
        cls,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> "QwenOllama":
        """Create a QwenOllama instance with the specified configuration.

        Args:
            model: The model name (e.g., "qwen2.5-coder:14b")
            base_url: The base URL for the Ollama server
            api_key: API key (dummy key for local servers)
            temperature: Temperature setting for the model
            **kwargs: Additional arguments to pass to ChatOpenAI

        Returns:
            A configured QwenOllama instance
        """
        return cls(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )
