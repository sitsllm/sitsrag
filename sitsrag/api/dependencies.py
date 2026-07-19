#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Request-scoped dependencies."""

from __future__ import annotations

from fastapi import Request

from sitsrag.api.sse.limiter import ConnectionLimiter


def get_connection_limiter(request: Request) -> ConnectionLimiter:
    """Return the connection limiter from app state.

    Args:
        request (Request): FastAPI request object.

    Returns:
        ConnectionLimiter: Connection limiter.
    """
    return request.app.state.connection_limiter
