#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the AG-UI endpoint and `sources_node`."""

import types

import httpx
from langchain_core.messages import ToolMessage

from sitsrag.api.agui.router import _sanitize_tool_call_args
from sitsrag.services.agent import AgentState, sources_node

#
# AG-UI payload
#
AGUI_PAYLOAD = {
    "threadId": "4e2bcb7e-157c-4b4c-af3b-9ed37c496b6f",
    "runId": "r1",
    "state": {},
    "messages": [],
    "tools": [],
    "context": [],
    "forwardedProps": {},
}


#
# Auxiliary functions
#
def _assistant_msg(*arguments: str):
    """Build a duck-typed assistant message with tool calls carrying raw arguments.

    Args:
        *arguments (str): The arguments to build the message with.

    Returns:
        types.SimpleNamespace: The built message.
    """
    # Build tool calls
    tool_calls = [
        types.SimpleNamespace(function=types.SimpleNamespace(arguments=a)) for a in arguments
    ]

    # Return!
    return types.SimpleNamespace(tool_calls=tool_calls)


#
# AG-UI endpoint (consolidated)
#
async def test_agui_endpoint_streams_events(test_app):
    """Test that the AG-UI endpoint returns a 200 SSE with `RUN_STARTED` and `RUN_FINISHED` events."""
    # Build transport
    transport = httpx.ASGITransport(app=test_app)

    # Send request
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/agui/agent", json=AGUI_PAYLOAD)

    # Assert response
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "RUN_STARTED" in response.text
    assert "RUN_FINISHED" in response.text


async def test_agui_valid_thread_id_accepted(test_app):
    """Test that the AG-UI endpoint accepts a valid UUID thread_id."""
    # Build transport
    transport = httpx.ASGITransport(app=test_app)

    # Send request
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            url="/agui/agent",
            json=AGUI_PAYLOAD,
        )

    # Assert response
    assert response.status_code == 200


async def test_agui_invalid_thread_id_returns_400(test_app):
    """Test that the AG-UI endpoint rejects non-UUID thread_id with 400."""
    # Build payload
    payload = {**AGUI_PAYLOAD, "threadId": "not-a-uuid"}

    # Build transport
    transport = httpx.ASGITransport(app=test_app)

    # Send request
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            url="/agui/agent",
            json=payload,
        )

    # Assert response
    assert response.status_code == 400
    assert "thread_id" in response.json()["detail"].lower()


#
# Tool-call argument sanitizer
#
def test_sanitize_repairs_concatenated_arguments():
    """Two glued JSON objects collapse to the first valid one."""
    # Build message
    msg = _assistant_msg('{"query": "a"}{"query": "b"}')

    # Sanitize tool call arguments
    _sanitize_tool_call_args([msg])

    # Assert arguments
    assert msg.tool_calls[0].function.arguments == '{"query": "a"}'


def test_sanitize_leaves_valid_arguments_untouched():
    """A single well-formed object is preserved (re-serialized identically)."""
    # Build message
    msg = _assistant_msg('{"query": "a"}')

    # Sanitize tool call arguments
    _sanitize_tool_call_args([msg])

    # Assert arguments
    assert msg.tool_calls[0].function.arguments == '{"query": "a"}'


def test_sanitize_replaces_unparseable_arguments_with_empty_object():
    """Garbage that cannot be decoded becomes an empty object, not a crash."""
    # Build message
    msg = _assistant_msg("not json at all")

    # Sanitize tool call arguments
    _sanitize_tool_call_args([msg])

    # Assert arguments
    assert msg.tool_calls[0].function.arguments == "{}"


def test_sanitize_handles_empty_and_missing():
    """Empty/blank args and messages without tool_calls are no-ops."""
    # Build message
    msg = _assistant_msg("", "   ")

    # Sanitize tool call arguments
    _sanitize_tool_call_args([msg, types.SimpleNamespace()])

    # Assert arguments
    assert msg.tool_calls[0].function.arguments == ""
    assert msg.tool_calls[1].function.arguments == "   "

    # Sanitize tool call arguments
    _sanitize_tool_call_args(None)  # must not raise


#
# Sources node
#
async def test_sources_node_collects_artifact_sources():
    """Test that the sources node collects artifact sources."""
    # Build message
    msg = ToolMessage(
        content="[S1] Title A",
        tool_call_id="call_1",
        artifact={
            "sources": {
                "S1": {"title": "Title A", "url": "https://a.com", "source_type": "documentation"}
            }
        },
    )

    # Build state
    state = AgentState(messages=[msg])
    result = await sources_node(state, config={})

    # Assert result
    assert result["sources"]["S1"]["title"] == "Title A"
    assert result["sources"]["S1"]["url"] == "https://a.com"


async def test_sources_node_skips_messages_without_artifact():
    """Test that the sources node skips messages without artifact."""
    # Build message
    msg = ToolMessage(
        content="No results found.",
        tool_call_id="call_1",
    )

    # Build state
    state = AgentState(messages=[msg])

    # Run sources node
    result = await sources_node(state, config={})

    # Assert result
    assert result == {"sources": {}}


async def test_sources_node_merges_multiple_tool_artifacts():
    """Test that the sources node merges multiple tool artifacts."""
    # Build message
    msg1 = ToolMessage(
        content="[S1] Doc",
        tool_call_id="call_1",
        artifact={"sources": {"S1": {"title": "Doc", "url": None, "source_type": "documentation"}}},
    )

    # Build message
    msg2 = ToolMessage(
        content="[S1] Ref",
        tool_call_id="call_2",
        artifact={
            "sources": {
                "S1": {"title": "Ref", "url": "https://ref.com", "source_type": "reference"}
            }
        },
    )

    # Build state
    state = AgentState(messages=[msg1, msg2])

    # Run sources node
    result = await sources_node(state, config={})

    # Assert result
    assert len(result["sources"]) == 1
    assert result["sources"]["S1"]["title"] == "Ref"
