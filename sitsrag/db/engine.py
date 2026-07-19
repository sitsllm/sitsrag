#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""SQLite database engine."""

from __future__ import annotations

import sqlite_vec
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

#
# Import data models
#
import sitsrag.db.ontology.models as ontology_models  # noqa: F401
from sitsrag.db.base import Base


def _register_sqlite_vec(engine: AsyncEngine) -> None:
    """Register the sqlite-vec extension."""

    @event.listens_for(engine.sync_engine, "connect")
    def _load_extension(dbapi_conn, _record) -> None:
        # dbapi_conn is SQLAlchemy's async adapter. Uses the raw sqlite3 conn
        # (aiosqlite.Connection._conn) to call the synchronous extension API.
        raw = dbapi_conn.driver_connection._conn
        raw.enable_load_extension(True)

        # Load sqlite-vec extension
        sqlite_vec.load(raw)
        raw.enable_load_extension(False)

        # WAL improves concurrent read/write behaviour
        raw.execute("PRAGMA journal_mode=WAL")


def create_engine(database_url: str, *, load_vec: bool = False) -> AsyncEngine:
    """Create a SQLAlchemy async engine backed by SQLite.

    Args:
        database_url (str): ``sqlite+aiosqlite:///path`` connection string.

        load_vec (bool): When True, load the sqlite-vec extension on each connection.

    Returns:
        AsyncEngine: Configured async engine.
    """
    engine = create_async_engine(database_url)

    if load_vec:
        _register_sqlite_vec(engine)

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory.

    Args:
        engine (AsyncEngine): SQLAlchemy engine.

    Returns:
        async_sessionmaker[AsyncSession]: Session factory.
    """
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )


async def init_db(engine: AsyncEngine) -> None:
    """Create all ORM tables.

    Args:
        engine (AsyncEngine): SQLAlchemy engine.

    Returns:
        None
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
