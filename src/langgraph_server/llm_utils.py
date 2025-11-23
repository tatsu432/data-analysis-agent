"""Utilities for initializing Language Models (LLMs).

This module provides a centralized function for instantiating chat models
based on configuration settings.
"""

import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .llm_wrappers import QwenOllama
from .settings import LLMProvider, LLMSettings, OptionalLLMSettings

logger = logging.getLogger(__name__)


def initialize_llm(
    config: LLMSettings, fallback_llm_config: OptionalLLMSettings = None
) -> BaseChatModel:
    """Initializes and returns a chat model based on the provided LLM settings.

    This function configures and instantiates a chat model using the
    attributes of the given `config` object. It utilizes the `init_chat_model`
    utility from LangChain to create the appropriate model instance based on
    the configured provider (OpenAI, Anthropic, Local, or QwenOllama).

    The specific model name, API keys, temperature, and other parameters
    are sourced from the `config.llm_params` property.

    Args:
        config: A `LLMSettings` object containing the configuration
            for the LLM to be initialized.
        fallback_llm_config: Optional fallback LLM configuration to use
            if the primary LLM fails.

    Returns:
        BaseChatModel: An instance of the configured chat model (e.g.,
        ChatOpenAI, ChatAnthropic, QwenOllama, or local OpenAI-compatible model).

    """
    logger.info(
        f"Initializing LLM: Model Name='{config.llm_model_name}', "
        f"Provider='{config.llm_model_provider}', "
        f"Temperature={config.temperature}"
    )

    try:
        # Special handling for QwenOllama provider
        if config.llm_model_provider == LLMProvider.QWEN_OLLAMA:
            llm_params = config.llm_params
            logger.info(
                f"Using QwenOllama wrapper for LangSmith visibility (model: {config.llm_model_name})"
            )
            primary = QwenOllama.create(
                model=config.llm_model_name,
                base_url=llm_params.get("base_url", "http://localhost:11434/v1"),
                api_key=llm_params.get("api_key", "ollama"),
                temperature=config.temperature,
            )
        else:
            # For other providers (OpenAI, Anthropic, Local), use standard init_chat_model
            # For local LLMs, llm_params contains model_provider="openai" and base_url
            # Extract model_provider from llm_params if present, otherwise use config.llm_model_provider
            llm_params = config.llm_params.copy()
            model_provider = llm_params.pop("model_provider", config.llm_model_provider)

            primary = init_chat_model(
                model=config.llm_model_name,
                model_provider=model_provider,
                temperature=config.temperature,
                **llm_params,
            )
        if fallback_llm_config:
            # Handle fallback LLM (same logic as primary)
            if fallback_llm_config.llm_model_provider == LLMProvider.QWEN_OLLAMA:
                fallback_llm_params = fallback_llm_config.llm_params
                logger.info(
                    f"Using QwenOllama wrapper for fallback LLM (model: {fallback_llm_config.llm_model_name})"
                )
                fallback_llm = QwenOllama.create(
                    model=fallback_llm_config.llm_model_name,
                    base_url=fallback_llm_params.get(
                        "base_url", "http://localhost:11434/v1"
                    ),
                    api_key=fallback_llm_params.get("api_key", "ollama"),
                    temperature=fallback_llm_config.temperature,
                )
            else:
                fallback_llm_params = fallback_llm_config.llm_params.copy()
                fallback_model_provider = fallback_llm_params.pop(
                    "model_provider", fallback_llm_config.llm_model_provider
                )
                fallback_llm = init_chat_model(
                    model=fallback_llm_config.llm_model_name,
                    model_provider=fallback_model_provider,
                    temperature=fallback_llm_config.temperature,
                    **fallback_llm_params,
                )
            llm = primary.with_fallbacks([fallback_llm])
        else:
            llm = primary
        logger.info("LLM initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {e!s}", exc_info=True)
        raise
