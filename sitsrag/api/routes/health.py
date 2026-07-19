#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Service health."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from sitsrag.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Return service health status with diagnostics.

    Args:
        request (Request): FastAPI request object.

    Notes:
        - Includes SSE connection metrics, database connectivity, and
          Agent Protocol resource counts when available.

    Returns:
        dict[str, Any]: Health status and diagnostics.
    """
    result = {"status": "ok"}

    # SSE connection metrics
    limiter = getattr(request.app.state, "connection_limiter", None)

    if limiter is not None:
        result["sse"] = {
            "active_connections": limiter.active_connections,
            "max_connections": limiter.max_connections,
            "capacity_remaining": limiter.max_connections - limiter.active_connections,
        }

    return result
