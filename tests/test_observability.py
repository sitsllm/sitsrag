#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the Langfuse (SDK v4) observability integration."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import litellm
import pytest

from sitsrag.config import Settings
from sitsrag.observability import (
    RUN_OBSERVATION_NAME,
    RunTrace,
    build_trace_run,
    build_trace_span,
    configure_langfuse,
)


#
# Auxiliary functions
#
def _fake_client(root: MagicMock | None = None, fail: bool = False) -> MagicMock:
    """Build a fake Langfuse client.

    Args:
        root (MagicMock | None): Root observation.

        fail (bool): Raise when starting an observation.

    Returns:
        MagicMock: Fake client.
    """

    client = MagicMock()
    entered = []
    client.entered = entered

    @contextmanager
    def _observation(**kwargs):
        if fail:
            raise RuntimeError("langfuse down")

        entered.append("enter")

        try:
            yield root if root is not None else MagicMock()

        finally:
            entered.append("exit")

    # Patch observation context manager
    client.start_as_current_observation.side_effect = _observation

    # Return!
    return client


#
# configure_langfuse
#
def test_configure_langfuse_disabled_returns_none():
    """Test that a disabled Langfuse returns no client."""
    settings = Settings(
        langfuse_enabled=False,
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )

    assert configure_langfuse(settings) is None


def test_configure_langfuse_missing_keys_returns_none():
    """Test that missing keys return no client."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )

    assert configure_langfuse(settings) is None


def test_configure_langfuse_builds_v4_client(monkeypatch):
    """Test that the v4 client is built from settings."""
    langfuse = pytest.importorskip("langfuse")

    # Patch Langfuse constructor
    fake_ctor = MagicMock(return_value="client")
    monkeypatch.setattr(langfuse, "Langfuse", fake_ctor)

    # Configure
    settings = Settings(
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_host="https://langfuse.example.org",
        environment="staging",
    )

    client = configure_langfuse(settings)

    # Assert client
    assert client == "client"
    fake_ctor.assert_called_once_with(
        public_key="pk",
        secret_key="sk",
        base_url="https://langfuse.example.org",
        environment="staging",
    )

    # Assert callbacks
    assert "langfuse" not in litellm.success_callback
    assert "langfuse" not in litellm.failure_callback


#
# build_trace_span
#
async def test_trace_span_without_client_is_noop():
    """Test that `trace_span` is a no-op when no client is provided."""
    trace_span = build_trace_span(None)

    async with trace_span("tool:test", query="q"):
        pass


async def test_trace_span_uses_observation_api():
    """Test that `trace_span` opens a span observation."""
    client = _fake_client()
    trace_span = build_trace_span(client)

    async with trace_span("tool:test", query="q"):
        assert client.entered == ["enter"]

    # Assert span observation
    client.start_as_current_observation.assert_called_once_with(
        as_type="span",
        name="tool:test",
        metadata={"query": "q"},
    )

    # Assert entered
    assert client.entered == ["enter", "exit"]


async def test_trace_span_fails_open():
    """Test that a failing client does not break the traced code."""
    ran = False
    trace_span = build_trace_span(_fake_client(fail=True))

    async with trace_span("tool:test"):
        ran = True

    assert ran


async def test_trace_span_propagates_body_exceptions():
    """Test that exceptions raised inside the span propagate."""
    client = _fake_client()
    trace_span = build_trace_span(client)

    # Raise exception
    with pytest.raises(ValueError):
        async with trace_span("tool:test"):
            raise ValueError("boom")

    assert client.entered == ["enter", "exit"]


#
# build_trace_run
#
async def test_trace_run_without_client_is_noop():
    """Test that `trace_run` yields an inert run when no client is provided."""
    trace_run = build_trace_run(None)

    async with trace_run(thread_id="t1", run_id="r1", input="hi") as run:
        assert isinstance(run, RunTrace)
        assert run.callbacks == []

        # No-op
        run.set_output("done")


async def test_trace_run_builds_root_observation_and_callbacks(monkeypatch):
    """Test that `trace_run` opens a root observation and propagates the session."""
    # Patch propagate_attributes
    propagate_calls = []

    langfuse = pytest.importorskip("langfuse")
    langfuse_langchain = pytest.importorskip("langfuse.langchain")

    @contextmanager
    def _propagate(**kwargs):
        propagate_calls.append(kwargs)
        yield

    # Patch CallbackHandler
    handler = MagicMock(name="handler")
    monkeypatch.setattr(langfuse, "propagate_attributes", _propagate)
    monkeypatch.setattr(langfuse_langchain, "CallbackHandler", MagicMock(return_value=handler))

    # Fake client and root observation
    root = MagicMock()
    client = _fake_client(root=root)
    trace_run = build_trace_run(client)

    # Open run
    async with trace_run(thread_id="thread-1", run_id="run-1", input="hello") as run:
        assert run.callbacks == [handler]
        run.set_output("world")

    # Root observation carries the run input and is closed
    client.start_as_current_observation.assert_called_once_with(
        as_type="agent",
        name=RUN_OBSERVATION_NAME,
        input="hello",
        metadata={"run_id": "run-1"},
    )
    assert client.entered == ["enter", "exit"]

    # Output lands on the root observation
    root.update.assert_called_once_with(output="world")

    # Session is propagated to children
    assert propagate_calls == [
        {
            "session_id": "thread-1",
            "trace_name": RUN_OBSERVATION_NAME,
            "metadata": {"run_id": "run-1"},
        }
    ]


async def test_trace_run_fails_open():
    """Test that a failing client yields an inert run."""
    trace_run = build_trace_run(_fake_client(fail=True))

    async with trace_run(thread_id="t1") as run:
        assert run.callbacks == []
