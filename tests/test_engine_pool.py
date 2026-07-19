#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the SQLite engine + sqlite-vec loading."""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import text

from sitsrag.db.engine import create_engine


@pytest.mark.asyncio
async def test_engine_loads_sqlite_vec_extension():
    """An engine created with load_vec=True can call sqlite-vec functions."""
    # Build temporary database path
    tmp = Path(tempfile.mkdtemp()) / "i.db"

    # Create engine
    engine = create_engine(f"sqlite+aiosqlite:///{tmp}", load_vec=True)

    # Try to execute SQL
    try:
        async with engine.connect() as conn:
            row = (await conn.execute(text("select vec_length(vec_f32('[1,2,3]'))"))).fetchone()

        # Assert result
        assert row[0] == 3

    # Finally, dispose of the engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_engine_without_vec_has_no_extension():
    """Without load_vec, sqlite-vec functions are unavailable."""
    # Build temporary database path
    tmp = Path(tempfile.mkdtemp()) / "r.db"

    # Create engine
    engine = create_engine(f"sqlite+aiosqlite:///{tmp}")

    # Try to execute SQL
    try:
        async with engine.connect() as conn:
            with pytest.raises(Exception):
                await conn.execute(text("select vec_length(vec_f32('[1,2,3]'))"))

    finally:
        await engine.dispose()
