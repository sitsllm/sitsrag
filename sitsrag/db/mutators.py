#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Database-level mutation utilities."""

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase


async def mut_db_upsert_rows(
    session: AsyncSession,
    model: type[DeclarativeBase],
    rows: list[dict],
    key: str | list[str],
) -> int:
    """Upsert rows with ``ON CONFLICT DO UPDATE`` on the given key column(s).

    Args:
        session (AsyncSession): SQLAlchemy async session.

        model (type[DeclarativeBase]): SQLAlchemy model class.

        rows (list[dict]): List of dicts to upsert.

        key (str | list[str]): Column or list of columns to use as the key for the upsert.

    Returns:
        int: Number of rows processed.
    """
    # Build index elements
    index_elements = [key] if isinstance(key, str) else key

    # Upsert rows
    for row in rows:
        # Build insert statement
        stmt = (
            sqlite_insert(model)
            .values(**row)
            .on_conflict_do_update(index_elements=index_elements, set_=row)
        )

        # Execute statement
        await session.execute(stmt)

    # Return number of rows processed
    return len(rows)


async def mut_db_insert_ignore_rows(
    session: AsyncSession,
    model: type[DeclarativeBase],
    rows: list[dict],
) -> int:
    """Insert rows with ``ON CONFLICT DO NOTHING``.

    Args:
        session (AsyncSession): SQLAlchemy async session.

        model (type[DeclarativeBase]): SQLAlchemy model class.

        rows (list[dict]): List of dicts to insert.

    Returns:
        int: Number of rows processed.
    """
    # Insert rows
    for row in rows:
        # Build insert statement
        stmt = sqlite_insert(model).values(**row).on_conflict_do_nothing()

        # Execute statement
        await session.execute(stmt)

    # Return number of rows processed
    return len(rows)
