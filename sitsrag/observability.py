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

from collections.abc import Callable
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sitsrag.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from langchain_core.callbacks import BaseCallbackHandler
    from langfuse import Langfuse

    from sitsrag.config import Settings


#
# Logger
#
logger = get_logger(__name__)


#
# Constants
#
RUN_OBSERVATION_NAME = "agui.agent"

# Langfuse caps propagated attribute values (session id, metadata values)
ATTRIBUTE_MAX_LENGTH = 200


#
# Types
#
TraceSpan = Callable[..., "asynccontextmanager[None]"]
TraceRun = Callable[..., "asynccontextmanager[RunTrace]"]


@dataclass
class RunTrace:
    """Handle for one traced agent run.

    Attributes:
        callbacks (list[BaseCallbackHandler]): LangChain callbacks to attach to the
            graph run config. Empty when tracing is disabled.
    """

    _root: Any = None
    """The root observation for the run."""

    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    """Callbacks to attach to the run config."""

    #
    # Methods
    #
    def set_output(self, output: Any) -> None:
        """Record the run output on the root observation."""
        if self._root is None:
            return

        try:
            self._root.update(output=output)

        except Exception:
            logger.debug("Langfuse root output update failed")


#
# Configure Langfuse
#
def configure_langfuse(settings: Settings) -> Langfuse | None:
    """Configure the Langfuse client.

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

    # Create langfuse client
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_host,
        environment=settings.environment,
    )

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

        # Only guard the span setup: exceptions raised by the traced
        # body must propagate unchanged (re-yielding after a throw would break the generator)
        stack = ExitStack()

        try:
            stack.enter_context(
                client.start_as_current_observation(
                    as_type="span",
                    name=name,
                    metadata=attrs,
                )
            )

        except Exception:
            stack.close()
            logger.debug("Langfuse span creation failed", span_name=name)

        with stack:
            yield

    return trace_span


def build_trace_run(client: Langfuse | None) -> TraceRun:
    """Build a `trace_run` async context manager bound to a Langfuse client.

    Args:
        client (Langfuse | None): Langfuse client, or `None` to produce a no-op run.

    Returns:
        An async context manager factory for tracing agent runs.

    Notes:
        - Each run becomes one trace: a root observation carrying the run input/output,
        with the thread id propagated as session id to every child observation,
        and a LangChain callback handler that records the graph's LLM, and tool calls.
    """

    @asynccontextmanager
    async def trace_run(
        *,
        thread_id: str,
        run_id: str | None = None,
        input: Any = None,
    ) -> AsyncIterator[RunTrace]:
        """Trace an agent run."""
        if client is None:
            yield RunTrace()
            return

        # Only guard the trace setup
        stack = ExitStack()

        try:
            from langfuse import propagate_attributes
            from langfuse.langchain import CallbackHandler

            # Prepare metadata
            metadata = {"run_id": run_id[:ATTRIBUTE_MAX_LENGTH]} if run_id else None

            # Create the root observation
            root = stack.enter_context(
                client.start_as_current_observation(
                    as_type="agent",
                    name=RUN_OBSERVATION_NAME,
                    input=input,
                    metadata=metadata,
                )
            )

            # Propagate attributes to the root observation
            stack.enter_context(
                propagate_attributes(
                    session_id=thread_id[:ATTRIBUTE_MAX_LENGTH],
                    trace_name=RUN_OBSERVATION_NAME,
                    metadata=metadata,
                )
            )

            # Create the run trace
            run = RunTrace(callbacks=[CallbackHandler()], _root=root)

        except Exception:
            stack.close()
            logger.debug("Langfuse run trace failed")
            run = RunTrace()

        # Yield the run trace
        with stack:
            yield run

    return trace_run
