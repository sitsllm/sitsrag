# syntax=docker/dockerfile:1
#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

FROM python:3.12-slim

# uv for dependency management (pinned binary from the official image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Bake the local models into a stable cache path so workers start offline, and
# let uv compile bytecode / copy (not symlink) into the venv for a portable image.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install production dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra recommended

# Pre-download the local embedding model at build time so the first request does
# not pay a cold download (reranking is a remote Cohere call, nothing to bake).
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en')"

# Application code and the production gunicorn config, then install the project
COPY sitsrag/ sitsrag/
COPY gunicorn.conf.py ./
RUN uv sync --frozen --no-dev --extra recommended

# Run as a non-root user with read access to the venv and baked model cache
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app /opt/fastembed_cache
USER app

# Use the built virtualenv directly so startup does not re-sync via `uv run`
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://localhost:8000/api/health').raise_for_status()"]

CMD ["gunicorn", "-c", "gunicorn.conf.py", "sitsrag.asgi:app"]
