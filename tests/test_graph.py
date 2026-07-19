#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the agent graph builder."""

from unittest.mock import MagicMock

from langgraph.checkpoint.memory import MemorySaver

from sitsrag.services.agent import SYSTEM_PROMPT, build_agent_graph


def test_system_prompt_has_current_date_placeholder():
    """Test the system prompt has the current date placeholder."""
    assert "{current_date}" in SYSTEM_PROMPT


def test_build_agent_graph_returns_compiled_graph():
    """Test the build_agent_graph function returns a compiled graph."""
    # Build mock LLM
    mock_llm = MagicMock()

    # Build graph
    graph = build_agent_graph(
        llm=mock_llm,
        tools=[],
        checkpointer=MemorySaver(),
        system_prompt="You are a test assistant.",
    )

    # Assert result
    assert graph is not None
    assert hasattr(graph, "ainvoke")
