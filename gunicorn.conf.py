#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""SITS RAG - Gunicorn configuration for the production ASGI server."""

import os

# Bind on the container network (override with API_HOST / API_PORT).
bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8000')}"

# ASGI worker so gunicorn can serve the FastAPI app.
worker_class = "uvicorn_worker.UvicornWorker"

# SINGLE worker by default. The in-memory DailyQuota (services/quota.py), the
# MemorySaver checkpointer (main.py) and the per-process SSE ConnectionLimiter
# all assume one process. Raising this would split those counters per worker.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))

# SSE-friendly timeouts. Long enough for a streamed agent run to finish.
timeout = int(os.getenv("WORKER_TIMEOUT", "300"))
graceful_timeout = int(os.getenv("WORKER_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("KEEPALIVE", "5"))

# Trust forwarded headers from the nginx ingress (compose sets this to "*").
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1")

# Log to stdout/stderr for Docker.
accesslog = "-"
errorlog = "-"
