"""Utilities for initializing Language Models (LLMs).

This module provides a centralized function for instantiating chat models
based on configuration settings.
"""

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .settings import LLMProvider, LLMSettings

logger = logging.getLogger(__name__)


def initialize_llm(config: LLMSettings) -> BaseChatModel:
    """Initializes and returns a chat model based on the provided LLM settings.

    This function configures and instantiates a chat model using the
    attributes of the given `config` object. It utilizes the `init_chat_model`
    utility from LangChain to create the appropriate model instance based on
    the configured provider (OpenAI, Anthropic, Local, or QwenOllama via ChatOpenAI).

    The specific model name, API keys, temperature, and other parameters
    are sourced from the `config.llm_params` property.

    Args:
        config: A `LLMSettings` object containing the configuration
            for the LLM to be initialized.

    Returns:
        BaseChatModel: An instance of the configured chat model (e.g.,
        ChatOpenAI, ChatAnthropic, or local OpenAI-compatible model).

    """
    logger.info(
        f"Initializing LLM: Model Name='{config.llm_model_name}', "
        f"Provider='{config.llm_model_provider}', "
        f"Temperature={config.temperature}"
    )

    try:
        # For all providers, use init_chat_model
        # For local LLMs and QwenOllama, llm_params contains model_provider="openai" and base_url
        # Extract model_provider from llm_params if present, otherwise use config.llm_model_provider
        llm_params = config.llm_params.copy()
        model_provider = llm_params.pop("model_provider", config.llm_model_provider)

        # Handle JSON mode for Ollama (QwenOllama and Local with Ollama)
        # Ollama uses extra_body["format"] instead of response_format
        if config.llm_model_provider in (LLMProvider.QWEN_OLLAMA, LLMProvider.LOCAL):
            response_format = llm_params.pop("response_format", None)
            if response_format:
                # Convert response_format to extra_body["format"] for Ollama
                extra_body: dict[str, Any] = llm_params.pop("extra_body", {})
                extra_body["format"] = "json"
                llm_params["extra_body"] = extra_body

        primary = init_chat_model(
            model=config.llm_model_name,
            model_provider=model_provider,
            temperature=config.temperature,
            **llm_params,
        )
        logger.info("LLM initialized successfully.")
        return primary
    except Exception as e:
        logger.error(f"Error initializing LLM: {e!s}", exc_info=True)
        raise
