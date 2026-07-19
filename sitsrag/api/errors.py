#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Error response model and exception handler."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sitsrag.logging import get_logger

#
# Logger
#
logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    """Agent Protocol-compatible error envelope."""

    code: str | None = None
    """Error code."""

    message: str = None
    """Error message."""

    metadata: dict[str, Any] | None = None
    """Optional metadata."""


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Serialise ``HTTPException`` as an ``ErrorResponse`` JSON body.

    Registered on the FastAPI application so that all `HTTPException` raises
    across the codebase produce a uniform Agent Protocol error envelope rather
    than FastAPI's default `{"detail": ...}` shape.

    Args:
        request (Request): The incoming Starlette request (unused but required by FastAPI).

        exc (HTTPException): The `HTTPException` raised by a route handler or middleware.

    Returns:
        A ``JSONResponse`` whose body is in ``ErrorResponse`` shaped and
        whose status code matches the exception `status_code`.
    """
    detail = exc.detail

    # If detail is a dict, safe to extract code and message
    if isinstance(detail, dict):
        code = detail.get("code", None)

        # Get message
        message = str(detail.get("message", "An unexpected error occurred."))

        # Get metadata
        metadata = {k: v for k, v in detail.items() if k not in {"code", "message"}} or None

    else:
        code = None
        metadata = None
        message = str(detail) if detail else "An unexpected error occurred."

    logger.info(
        "http_exception",
        status_code=exc.status_code,
        code=code,
        message=message,
        path=str(request.url.path),
    )

    # Define body
    body = ErrorResponse(
        code=code,
        message=message,
        metadata=metadata,
    )

    # Define headers
    headers = dict(exc.headers) if exc.headers else None

    # Return!
    return JSONResponse(
        content=body.model_dump(
            exclude_none=True,
        ),
        status_code=exc.status_code,
        headers=headers,
    )
