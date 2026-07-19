#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""CLI commands for indexing and re-indexing content."""

import asyncio

import typer

from sitsrag.cli.infra import console, vector_scope

#
# Create indexing commands app
#
index_app = typer.Typer(
    help="Index content into the vector store.",
    no_args_is_help=True,
)

#
# Create recreate commands app
#
recreate_app = typer.Typer(
    help="Delete existing vectors and re-index from scratch.",
    no_args_is_help=True,
)

# Add recreate commands to indexing app
index_app.add_typer(recreate_app, name="recreate")


#
# Index documentation command
#
@index_app.command("documentation")
def index_documentation() -> None:
    """Index the SITS book documentation."""

    # Define async function
    async def _run() -> int:

        # Run vector scope
        async with vector_scope() as (_settings, _engine, _vs, service):
            # Run indexing
            with console.status("[bold blue]Indexing SITS book documentation...", spinner="dots"):
                return await service.index_documentation()

    # Run async function
    count = asyncio.run(_run())

    # Inform user
    console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] chunks.")


#
# Index reference command
#
@index_app.command("reference")
def index_reference() -> None:
    """Index the SITS function reference."""

    # Define async function
    async def _run() -> int:

        # Run vector scope
        async with vector_scope() as (_settings, _engine, _vs, service):
            # Run indexing
            with console.status("[bold blue]Indexing SITS function reference...", spinner="dots"):
                return await service.index_reference()

    # Run async function
    count = asyncio.run(_run())

    # Inform user
    console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] chunks.")


#
# Index articles command
#
@index_app.command("articles")
def index_articles(
    path: str = typer.Option(None, "--path", help="Override the default articles JSON path."),
) -> None:
    """Index the curated catalog of articles using SITS."""

    # Define async function
    async def _run() -> int:

        # Run vector scope
        async with vector_scope() as (_settings, _engine, _vs, service):
            # Run indexing
            with console.status("[bold blue]Indexing SITS article catalog...", spinner="dots"):
                return await service.index_articles(path)

    # Run async function
    count = asyncio.run(_run())

    # Inform user
    console.print(f"[bold green]Done![/] Indexed [cyan]{count}[/] chunks.")


#
# Recreate documentation command
#
@recreate_app.command("documentation")
def recreate_documentation() -> None:
    """Delete existing documentation vectors and re-index from scratch."""

    # Define async function
    async def _run() -> int:

        # Run vector scope
        async with vector_scope() as (_settings, _engine, _vs, service):
            # Run re-indexing
            with console.status(
                "[bold yellow]Re-indexing SITS book documentation...",
                spinner="dots",
            ):
                return await service.reindex_documentation()

    # Run async function
    count = asyncio.run(_run())

    # Inform user
    console.print(f"[bold green]Done![/] Re-indexed [cyan]{count}[/] chunks.")


#
# Recreate reference command
#
@recreate_app.command("reference")
def recreate_reference() -> None:
    """Delete existing reference vectors and re-index from scratch."""

    # Define async function
    async def _run() -> int:

        # Run vector scope
        async with vector_scope() as (_settings, _engine, _vs, service):
            # Run re-indexing
            with console.status(
                "[bold yellow]Re-indexing SITS function reference...",
                spinner="dots",
            ):
                return await service.reindex_reference()

    # Run async function
    count = asyncio.run(_run())

    # Inform user
    console.print(f"[bold green]Done![/] Re-indexed [cyan]{count}[/] chunks.")


#
# Recreate articles command
#
@recreate_app.command("articles")
def recreate_articles(
    path: str = typer.Option(None, "--path", help="Override the default articles JSON path."),
) -> None:
    """Delete existing article vectors and re-index from scratch."""

    # Define async function
    async def _run() -> int:

        # Run vector scope
        async with vector_scope() as (_settings, _engine, _vs, service):
            # Run re-indexing
            with console.status(
                "[bold yellow]Re-indexing SITS article catalog...",
                spinner="dots",
            ):
                return await service.reindex_articles(path)

    # Run async function
    count = asyncio.run(_run())

    # Inform user
    console.print(f"[bold green]Done![/] Re-indexed [cyan]{count}[/] chunks.")
