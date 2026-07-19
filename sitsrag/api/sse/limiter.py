#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""SSE connection concurrency limiter."""

from __future__ import annotations

import asyncio
from types import TracebackType

from fastapi import HTTPException

from sitsrag.logging import get_logger

logger = get_logger(__name__)


class ConnectionLimiter:
    """Limit the number of concurrent SSE connections.

    When the configured ceiling is reached, ``acquire()`` raises an
    ``HTTP 503`` rather than blocking, so callers fail-fast instead of
    queuing behind a saturated server.

    Args:
        max_connections (int): Maximum number of simultaneous connections allowed.
    """

    def __init__(self, max_connections: int = 200) -> None:
        """Initializer.

        Args:
            max_connections (int): Maximum number of simultaneous connections allowed.

        Returns:
            None
        """
        self._max_connections = max_connections
        self._semaphore = asyncio.Semaphore(max_connections)

    @property
    def max_connections(self) -> int:
        """Return the configured connection ceiling.

        Returns:
            int: Maximum number of simultaneous connections allowed.
        """
        return self._max_connections

    @property
    def active_connections(self) -> int:
        """Return the number of currently active connections.

        Returns:
            int: Number of currently active connections.
        """
        return self._max_connections - self._semaphore._value  # type: ignore[attr-defined]

    async def acquire(self) -> None:
        """Acquire a connection slot.

        Returns:
            None

        Raises:
            HTTPException: HTTP 503 when the connection limit is reached.
        """
        if self._semaphore._value == 0:
            # Log
            logger.warning(
                "sse.connection_limit_reached",
                max_connections=self._max_connections,
                active_connections=self.active_connections,
            )

            # Raise HTTP exception
            raise HTTPException(
                status_code=503,
                detail="Too many concurrent connections. Please retry later.",
            )

        # Acquire semaphore
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release a previously acquired connection slot.

        Returns:
            None
        """
        # Release semaphore
        self._semaphore.release()

    async def __aenter__(self) -> ConnectionLimiter:
        """Acquire on context entry.

        Returns:
            ConnectionLimiter: Connection limiter.
        """
        # Acquire semaphore
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release on context exit regardless of exception.

        Args:
            exc_type (type[BaseException] | None): Exception type.

            exc_val (BaseException | None): Exception value.

            exc_tb (TracebackType | None): Exception traceback.

        Returns:
            None
        """
        self.release()
