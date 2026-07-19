#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Observability utilities."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import litellm

from sitsrag.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from langfuse import Langfuse

    from sitsrag.config import Settings


#
# Logger
#
logger = get_logger(__name__)


#
# Types
#
TraceSpan = Callable[..., "asynccontextmanager[None]"]


#
# Configure Langfuse
#
def configure_langfuse(settings: Settings) -> Langfuse | None:
    """Configure Langfuse observability via LiteLLM callbacks.

    Args:
        settings (Settings): Configuration.

    Returns:
        A Langfuse client if configured, `None` otherwise.
    """
    # Check if Langfuse is enabled
    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled via config")
        return None

    # Check if Langfuse keys are configured
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info("Langfuse keys not configured - observability disabled")
        return None

    # Create and return the client (imported here as this is optional)
    from langfuse import Langfuse

    # LiteLLM's Langfuse callback reads API keys from env vars - there is
    # no programmatic alternative for the callback system. This is the one
    # place in the codebase where os.environ mutation is intentional.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    os.environ.setdefault("LANGFUSE_RELEASE", settings.environment)

    # Register LiteLLM callback
    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")

    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback.append("langfuse")

    # Create langfuse client
    client = Langfuse()

    # Log
    logger.info(
        event="Langfuse observability enabled",
        host=settings.langfuse_host,
        environment=settings.environment,
    )

    # Return!
    return client


def build_trace_span(client: Langfuse | None) -> TraceSpan:
    """Build a `trace_span` async context manager bound to a Langfuse client.

    Args:
        client (Langfuse | None): Langfuse client, or `None` to produce a no-op span.

    Returns:
        An async context manager factory for tracing spans.
    """

    @asynccontextmanager
    async def trace_span(name: str, **attrs: Any) -> AsyncIterator[None]:
        if client is None:
            yield
            return

        try:
            span = client.span(name=name, metadata=attrs)

            try:
                yield

            finally:
                span.end()

        except Exception:
            logger.debug("Langfuse span failed — continuing without trace", span_name=name)
            yield

    return trace_span
