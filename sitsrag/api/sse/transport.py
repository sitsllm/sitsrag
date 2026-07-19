#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""SSE transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

from sitsrag.logging import get_logger

logger = get_logger(__name__)


class SSETransport:
    """Server-Sent Events transport.

    Implements a producer-consumer pattern where producers call ``send()``
    and the HTTP handler iterates ``stream()``.  The queue is bounded so
    that a slow consumer exerts backpressure on producers rather than
    accumulating unbounded memory.
    """

    def __init__(
        self,
        heartbeat_interval: float = 15.0,
        max_queue_size: int = 256,
    ) -> None:
        """Initializer.

        Args:
            heartbeat_interval (float): Seconds between keep-alive heartbeat comments.
            max_queue_size (int): Maximum number of pending SSE frames in the queue.

        Returns:
            None
        """
        self._disconnected = False
        self._heartbeat_interval = heartbeat_interval
        self._queue = asyncio.Queue(maxsize=max_queue_size)

    #
    # Properties
    #
    @property
    def disconnected(self) -> bool:
        """Return ``True`` after the client has disconnected.

        Returns:
            bool: ``True`` if the client has disconnected, ``False`` otherwise.
        """
        return self._disconnected

    async def send(self, event: str, data: str) -> None:
        """Enqueue an SSE event frame.

        Args:
            event (str): SSE event name (``event:`` field).

            data (str): Payload string (``data:`` field).

        Notes:
            - Silently drops the frame when the client has already disconnected to
            avoid blocking indefinitely on a dead connection.

        Returns:
            None
        """
        if self._disconnected:
            return

        # Define event frame
        frame = f"event: {event}\ndata: {data}\n\n"

        await self._queue.put(frame)

    async def send_comment(self, comment: str) -> None:
        """Enqueue an SSE comment frame.

        Args:
            comment (str): Comment text (must not contain newlines).

        Returns:
            None
        """
        if self._disconnected:
            return

        # Define comment frame
        frame = f": {comment}\n\n"

        await self._queue.put(frame)

    async def send_error(self, code: str, message: str) -> None:
        """Enqueue a structured ``error`` event.

        Args:
            code (str): Machine-readable error code.

            message (str): Human-readable error description.

        Returns:
            None
        """
        # Encode payload
        payload = json.dumps({"code": code, "message": message})

        # Send error event
        await self.send("error", payload)

    async def close(self) -> None:
        """Close transport.

        Returns:
            None
        """
        await self._queue.put(None)

    async def stream(self) -> AsyncGenerator[str, None]:
        """Consume the queue and yield SSE frames.

        Yields a ``: heartbeat`` comment whenever no frame arrives within
        ``heartbeat_interval`` seconds, keeping the TCP connection alive
        through proxies and load balancers.

        Sets ``_disconnected = True`` in the ``finally`` block so that
        subsequent ``send()`` calls are silently discarded after the client
        disconnects or the generator is cancelled by Starlette.
        """
        get_task = None

        frames_sent = 0
        heartbeats_sent = 0

        logger.debug("stream.opened")

        try:
            while True:
                # If get task is None, get one from the queue
                if get_task is None:
                    get_task = asyncio.ensure_future(self._queue.get())

                # Wait for item from the queue
                try:
                    item = await asyncio.wait_for(
                        fut=asyncio.shield(get_task),
                        timeout=self._heartbeat_interval,
                    )

                # If timeout, send heartbeat and continue
                except TimeoutError:
                    heartbeats_sent += 1

                    logger.debug("stream.heartbeat", count=heartbeats_sent)

                    # Yield heartbeat comment
                    yield ": heartbeat\n\n"
                    continue

                # Reset get task
                get_task = None

                # If item is None, break
                if item is None:
                    return

                # Increment frames sent
                frames_sent += 1

                # Yield item
                yield item

        # Set disconnected flag and cancel get task
        finally:
            self._disconnected = True

            if get_task is not None and not get_task.done():
                get_task.cancel()

                try:
                    await get_task

                except (asyncio.CancelledError, Exception):
                    pass

            logger.info(
                "stream.closed",
                frames_sent=frames_sent,
                heartbeats_sent=heartbeats_sent,
            )


def sse_response(
    transport: SSETransport,
    media_type: str = "text/event-stream",
) -> StreamingResponse:
    """Create a StreamingResponse with SSE headers.

    Args:
        transport (SSETransport): The ``SSETransport`` whose ``stream()`` will be consumed.

        media_type (str): Content-Type header value.

    Returns:
        A ``StreamingResponse`` configured for SSE delivery.
    """
    return StreamingResponse(
        transport.stream(),
        media_type=media_type,
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
