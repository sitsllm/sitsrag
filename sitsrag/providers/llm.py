#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""LLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from sitsrag.config import Settings

#
# Global settings
#
litellm.drop_params = True


#
# Utilities
#
def _is_anthropic(model: str) -> bool:
    """Check whether a model name routes to the Anthropic provider.

    Args:
        model (str): The model name to check.

    Returns:
        bool: True if the model name routes to the Anthropic provider, False otherwise.
    """
    return model.startswith("anthropic/") or model.startswith("claude-")


def _is_reasoning_model(model: str) -> bool:
    """Check whether a model is a reasoning/thinking model.

    Args:
        model (str): The model name to check.

    Returns:
        bool: True if the model is a reasoning/thinking model, False otherwise.
    """
    lower = model.lower()
    return any(tag in lower for tag in ("o1", "o3"))


def _resolve_api_key(settings: Settings, model: str) -> dict[str, str]:
    """Return the provider-specific API key kwarg for ChatLiteLLM.

    Args:
        settings (Settings): Application settings.

        model (str): The model name to check.

    Returns:
        dict[str, str]: The provider-specific API key kwarg for ChatLiteLLM.
    """
    return {
        "anthropic_api_key": settings.chat_api_key or "",
    }


#
# High-level interface
#
def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build a LangChain chat model backed by LiteLLM.

    Reads all configuration from ``settings``: model name, API keys,
    temperature, and reasoning effort.

    Provider is detected automatically from the model name:
    - ``claude-*`` -> Anthropic
    - everything else -> OpenAI

    Args:
        settings (Settings): Application settings.

    Returns:
        BaseChatModel: A LangChain-compatible chat model.
    """
    model = settings.chat_model
    reasoning = _is_reasoning_model(model)

    # Use reasoning max tokens if the model is a reasoning model,
    # otherwise use the regular max tokens
    max_tokens = settings.llm_reasoning_max_tokens if reasoning else settings.llm_max_tokens

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "streaming": True,
        **_resolve_api_key(settings, model),
    }

    # Anthropic can emit parallel tool calls in one assistant turn. The AG-UI
    # streaming layer merges them into one buffer, producing invalid concatenated JSON.
    # Force sequential tool calls (ReAct works fine with such setup).
    if _is_anthropic(model):
        kwargs["model_kwargs"] = {"parallel_tool_calls": False}

    # Skip temperature for Anthropic and reasoning models (they ignore or error on it)
    if settings.llm_temperature is not None and not _is_anthropic(model) and not reasoning:
        kwargs["temperature"] = settings.llm_temperature

    # Add reasoning_effort for o1/o3 (LiteLLM drops it if unsupported)
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort

    return ChatLiteLLM(**kwargs)
