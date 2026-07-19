#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Shared test fixtures and helpers."""

from unittest.mock import MagicMock

import pytest
from ag_ui.core.events import (
    EventType,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from fastapi import FastAPI

from sitsrag.api.agui.router import router as agui_router
from sitsrag.api.routes.health import router as health_router
from sitsrag.api.sse.limiter import ConnectionLimiter


#
# Fixtures
#
@pytest.fixture()
def mock_agent():
    """Make a mock `LangGraphAGUIAgent` that streams a minimal AG-UI event sequence."""
    agent = MagicMock()
    agent.name = "test_agent"

    async def _run_side_effect(input_data):
        # Yield run started event
        yield RunStartedEvent(
            type=EventType.RUN_STARTED,
            run_id="r1",
            thread_id="t1",
        )

        # Yield text message start event
        yield TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id="m1",
            role="assistant",
        )

        # Yield text message content event
        yield TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id="m1",
            delta="Hello!",
        )

        # Yield text message end event
        yield TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id="m1",
        )

        # Yield run finished event
        yield RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            run_id="r1",
            thread_id="t1",
        )

    # Clone agent
    cloned = MagicMock()
    cloned.name = "test_agent"
    cloned.config = {}
    cloned.run = _run_side_effect

    agent.clone.return_value = cloned

    # Return!
    return agent


@pytest.fixture()
def test_app(mock_agent):
    """FastAPI test app with the AG-UI + health routers and mocked dependencies.

    Args:
        mock_agent (MagicMock): Mock agent.

    Returns:
        FastAPI: Test app.
    """
    # Build app
    app = FastAPI()
    app.include_router(router=agui_router)
    app.include_router(
        router=health_router,
        prefix="/api",
        tags=["health"],
    )

    # Set app state
    app.state.agent = mock_agent
    app.state.connection_limiter = ConnectionLimiter(max_connections=10)
    app.state.agent_max_iterations = 15
    app.state.sse_heartbeat_interval = 15.0
    app.state.sse_max_queue_size = 256
    app.state.agent_graph = MagicMock()

    # Return!
    return app
