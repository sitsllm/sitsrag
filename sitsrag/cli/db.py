#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Database management CLI commands."""

import asyncio

import typer

from sitsrag.cli.infra import console, engine_scope
from sitsrag.db.engine import init_db

#
# Create database commands app
#
db_app = typer.Typer(
    help="Database management commands.",
    no_args_is_help=True,
)


#
# Migrate command
#
@db_app.command("migrate")
def migrate() -> None:
    """Create all database tables (idempotent)."""

    # Define async function
    async def _run() -> None:

        # Run engine scope
        async with engine_scope() as (_settings, engine):
            # Run migrations
            with console.status("[bold blue]Running database migrations...", spinner="dots"):
                await init_db(engine)

            # Print success message
            console.print("[bold green]Done![/] Database tables created successfully.")

    # Run async function
    asyncio.run(_run())
