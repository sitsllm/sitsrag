#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""sits-rag - ASGI entrypoint for production (gunicorn/uvicorn)."""

from sitsrag.main import app

#
# ASGI application served by gunicorn (see gunicorn.conf.py)
#
__all__ = ("app",)
