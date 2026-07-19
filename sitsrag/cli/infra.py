#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Shared CLI infrastructure - context managers and console."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from rich.console import Console

from sitsrag.config import Settings
from sitsrag.db.engine import create_engine, create_session_factory, init_db

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

#
# Create console
#
console = Console()


#
# Create engine scope
#
@asynccontextmanager
async def engine_scope(
    settings: Settings | None = None,
) -> AsyncIterator[tuple]:
    """Yield engine scope.

    Args:
        settings: Optional pre-built settings. Created if omitted.

    Yields:
        Tuple of ``(settings, engine)``.
    """
    # Load app settings
    settings = settings or Settings()

    # Create engine (content index)
    engine = create_engine(settings.index_db_url, load_vec=True)

    # Yield engine
    try:
        yield settings, engine

    # Dispose engine
    finally:
        await engine.dispose()


#
# Create session scope
#
@asynccontextmanager
async def session_scope(
    settings: Settings | None = None,
) -> AsyncIterator[tuple]:
    """Yield session scope.

    Runs ``init_db`` to ensure the schema exists before yielding.

    Args:
        settings: Optional pre-built settings. Created if omitted.

    Yields:
        Tuple of ``(settings, engine, session_factory)``.
    """
    # Load app settings
    settings = settings or Settings()

    # Create engine (content index)
    engine = create_engine(settings.index_db_url, load_vec=True)
    session_factory = create_session_factory(engine)

    # Initialize database
    await init_db(engine)

    # Yield session
    try:
        yield settings, engine, session_factory

    # Dispose engine
    finally:
        await engine.dispose()


#
# Create vector scope
#
@asynccontextmanager
async def vector_scope(
    settings: Settings | None = None,
) -> AsyncIterator[tuple]:
    """Yield vector scope.

    Args:
        settings: Optional pre-built settings. Created if omitted.

    Yields:
        Tuple of ``(settings, engine, vector_store, IndexingService)``.
    """
    from sitsrag.db.graph.vector_store import create_vector_store
    from sitsrag.services.indexing import IndexingService

    # Load app settings
    settings = settings or Settings()

    # Create engine (content index) with sqlite-vec loaded
    engine = create_engine(settings.index_db_url, load_vec=True)

    # Ensure ontology + chunk/vector/FTS schema exists
    await init_db(engine)

    # Create vector store
    vector_store = create_vector_store(engine, settings)
    await vector_store.ensure_schema()

    # Create indexing service
    service = IndexingService(vector_store, settings)

    # Yield vector scope
    try:
        yield settings, engine, vector_store, service

    # Dispose engine
    finally:
        await engine.dispose()
