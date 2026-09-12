#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""AG-UI router."""

from __future__ import annotations

import asyncio
import json
from typing import cast
from uuid import UUID, uuid4

from ag_ui.core.events import (
    BaseEvent,
    EventType,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from copilotkit import LangGraphAGUIAgent
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from sitsrag.api.sse.limiter import ConnectionLimiter
from sitsrag.api.sse.transport import SSETransport, sse_response
from sitsrag.logging import get_logger
from sitsrag.observability import RunTrace, TraceRun, build_trace_run

logger = get_logger(__name__)

router = APIRouter(prefix="/agui", tags=["agui"])


def _sanitize_tool_call_args(messages: object) -> None:
    """Repair concatenated tool-call arguments from clients."""
    decoder = json.JSONDecoder()

    # Get messages
    for message in cast("list[object]", messages) or []:
        # Get tool calls
        for tool_call in getattr(message, "tool_calls", None) or []:
            # Get function
            function = getattr(tool_call, "function", None)

            # Get arguments
            raw = getattr(function, "arguments", None)

            # If raw is not a string or is empty, continue
            if not isinstance(raw, str) or not raw.strip():
                continue

            # Try to decode the arguments
            try:
                obj, _ = decoder.raw_decode(raw.strip())
                function.arguments = json.dumps(obj)

            # If error, set arguments to empty object
            except ValueError:
                function.arguments = "{}"


def _last_user_message(messages: object) -> str | None:
    """Get latest user message.

    Args:
        messages (object): The messages.

    Returns:
        The content of the latest user message, or ``None`` if no user
        message is found.
    """
    # Iterate over messages in reverse order
    for message in reversed(messages or []):
        # If message is a user message
        if getattr(message, "role", None) == "user":
            # Get content
            content = getattr(message, "content", None)

            # If content is a string, return it
            return content if isinstance(content, str) else None

    # No user message found
    return None


def _stream_notice(
    request: Request,
    encoder: EventEncoder,
    connection_limiter: ConnectionLimiter,
    input_data: RunAgentInput,
    text: str,
) -> StreamingResponse:
    """Stream a single canned assistant message as a valid AG-UI run.

    Args:
        request (Request): The request object.

        encoder (EventEncoder): The event encoder.

        connection_limiter (ConnectionLimiter): The connection limiter.

        input_data (RunAgentInput): The input data.

        text (str): The text to stream.

    Returns:
        StreamingResponse: The streaming response.
    """
    # Create transport
    transport = SSETransport(
        heartbeat_interval=request.app.state.sse_heartbeat_interval,
        max_queue_size=request.app.state.sse_max_queue_size,
    )

    # Get run id
    run_id = getattr(input_data, "run_id", None) or uuid4().hex

    # Get thread id
    thread_id = getattr(input_data, "thread_id", None) or uuid4().hex

    # Get message id
    message_id = uuid4().hex

    # Create events
    events = [
        # Run started
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            run_id=run_id,
            thread_id=thread_id,
        ),
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role="assistant",
        ),
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=message_id,
            delta=text,
        ),
        TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id,
        ),
        RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            run_id=run_id,
            thread_id=thread_id,
        ),
    ]

    async def producer() -> None:
        try:
            for event in events:
                await transport._queue.put(encoder.encode(event))  # type: ignore[attr-defined]

        finally:
            await transport.close()
            connection_limiter.release()

    # Create task and run producer
    _ = asyncio.create_task(producer())

    # Return sse response
    return sse_response(
        transport=transport,
        media_type=encoder.get_content_type(),
    )


@router.post("/agent")
async def agent_endpoint(input_data: RunAgentInput, request: Request) -> StreamingResponse:
    """Run the AG-UI agent.

    Note:
        This is a re-implementation of the CopilotKit AG-UI agent endpoint to ensure
        re-connection, disconnect handling, and double-framing prevention.
    """
    # Get app state
    app_state = cast(object, request.app.state)

    # Get agent
    agent = cast(LangGraphAGUIAgent, getattr(app_state, "agent"))

    # Get connection limiter
    connection_limiter = cast(ConnectionLimiter, getattr(app_state, "connection_limiter"))

    # Get trace run (no-op when Langfuse is not configured)
    trace_run = cast("TraceRun | None", getattr(app_state, "trace_run", None))
    trace_run = trace_run or build_trace_run(None)

    # Acquire connection
    await connection_limiter.acquire()

    try:
        try:
            # Validate thread ID
            UUID(input_data.thread_id)

        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=400,
                detail="Valid thread_id (UUID) is required.",
            )

        # Get accept header
        accept_header = request.headers.get("accept", "text/event-stream")

        # Create encoder
        encoder = EventEncoder(accept=accept_header)

        # Get daily quota
        quota = getattr(app_state, "daily_quota", None)

        # If quota is available
        if quota is not None and not quota.allow():
            # Get daily limit notice
            notice = getattr(app_state, "daily_limit_notice", "Daily limit reached.")

            # Log daily limit reached
            logger.warning(
                event="agui.daily_limit_reached",
                count=quota.count,
                limit=quota.limit,
            )

            # Stream notice
            return _stream_notice(
                request=request,
                encoder=encoder,
                connection_limiter=connection_limiter,
                input_data=input_data,
                text=notice,
            )

        # Repair concatenated tool-call arguments from clients
        _sanitize_tool_call_args(getattr(input_data, "messages", None))

        # Clone agent
        request_agent = agent.clone()

        # Get config
        config = cast(dict[str, object], request_agent.config)

        # Cap agent iterations
        config["recursion_limit"] = cast(int, getattr(app_state, "agent_max_iterations"))

        # Create transport
        transport = SSETransport(
            heartbeat_interval=request.app.state.sse_heartbeat_interval,
            max_queue_size=request.app.state.sse_max_queue_size,
        )

        # Stream the agent run to the transport
        async def _stream_run(run: RunTrace) -> None:
            # Attach Langfuse callbacks to the graph run config
            config["callbacks"] = run.callbacks

            # Assistant text collected for the trace output
            output_parts = []

            try:
                async for event in request_agent.run(input_data):
                    # If client disconnected, log and break
                    if transport.disconnected:
                        logger.info("agui.client_disconnected_mid_stream")
                        break

                    # Skip MESSAGES_SNAPSHOT
                    #  > The client builds messages from TEXT_MESSAGE_* and TOOL_CALL_* streaming events.
                    #  > The snapshot would replace those incremental messages with server-side copies
                    #  > that have different IDs, breaking smooth streaming animation.
                    #  > In theory, we can maintain this, but for some reason, the
                    #  > assistant-ui components seems to behave well when this is present.
                    if getattr(event, "type", None) == "MESSAGES_SNAPSHOT":
                        continue

                    # Get base event
                    base_event = cast(BaseEvent, cast(object, event))

                    # Collect assistant text deltas
                    if base_event.type == EventType.TEXT_MESSAGE_CONTENT:
                        output_parts.append(cast(TextMessageContentEvent, base_event).delta)

                    # Encode event (as SSE frame)
                    encoded_frame = encoder.encode(base_event)

                    # The disconnection guard is replicated here to match the
                    # semantics of send() and ensure that the client is not
                    # double-framed (this breaks the annimation)
                    if not transport.disconnected:
                        await transport._queue.put(encoded_frame)  # type: ignore[attr-defined]

            # Record whatever was streamed (also on partial/failed runs)
            finally:
                run.set_output("".join(output_parts))

        # Async producer function
        async def producer() -> None:
            try:
                async with trace_run(
                    thread_id=input_data.thread_id,
                    run_id=getattr(input_data, "run_id", None),
                    input=_last_user_message(getattr(input_data, "messages", None)),
                ) as run:
                    # Stream the run to the transport
                    await _stream_run(run)

            # If exception, send error and release connection
            except Exception as exc:
                logger.exception("agui.agent_error", exc=str(exc))

                await transport.send_error("agent_error", str(exc))

            # Close transport and release connection otherwise, the connection
            # will be left in a hung state
            finally:
                await transport.close()
                connection_limiter.release()

        # Create task and run producer
        asyncio.create_task(producer())

        # Return SSE response
        return sse_response(
            transport=transport,
            media_type=encoder.get_content_type(),
        )

    except Exception:
        # In any case, release the connection
        connection_limiter.release()
        raise


@router.get("/agent/health")
async def agent_health(request: Request) -> dict[str, object]:
    """Return health status and the agent name.

    Returns:
        A dict with ``status`` and ``agent.name``
    """
    app_state = cast(object, request.app.state)

    # Get agent
    agent = cast(LangGraphAGUIAgent, getattr(app_state, "agent"))

    # Return status and agent name
    return {
        "status": "ok",
        "agent": {"name": agent.name},
    }
