#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""CLI commands for ontology management."""

import asyncio
from pathlib import Path

import typer
from sqlalchemy import delete

from sitsrag.cli.infra import console, session_scope
from sitsrag.db.ontology.models import (
    Band,
    BandType,
    CollectionBand,
    Domain,
    IndexBandType,
    IndexDomain,
    IndexPlatform,
    Platform,
    PlatformSatellite,
    Satellite,
    Sensor,
    SensorBand,
    SensorSatellite,
    SitsCollection,
    SpectralIndex,
)
from sitsrag.ontology.geosatdb import (
    load_geosatdb,
)
from sitsrag.ontology.linker import link_ontologies
from sitsrag.ontology.sits_collections import (
    load_sits_collections,
)
from sitsrag.ontology.spectral_indices import (
    load_spectral_indices,
)

#
# Create ontology commands app
#
ontology_app = typer.Typer(
    help="Load and manage ontology data (GEOSatDB, Spectral Indices, SITS Collections).",
    no_args_is_help=True,
)


#
# Create load command
#
@ontology_app.command("load")
def load(
    source: str = typer.Argument(
        ..., help="Ontology to load: 'geosatdb', 'spectral-indices', 'sits-collections', or 'all'."
    ),
    path: Path = typer.Option(
        None, "--path", help="Override default data path for the selected source."
    ),
) -> None:
    """Load ontology data into vector store."""
    # Define valid ontology sources
    valid = {
        "geosatdb",
        "spectral-indices",
        "sits-collections",
        "all",
    }

    # If source is invalid, just exit
    if source not in valid:
        console.print(
            f"[bold red]Invalid source '{source}'. Choose from: {', '.join(sorted(valid))}[/]"
        )

        raise typer.Exit(code=1)

    # Define async function
    async def _run() -> None:
        # Run session scope
        async with session_scope() as (settings, _engine, session_factory):
            # If source is GEOSatDB or all, load GEOSatDB
            if source in {"geosatdb", "all"}:
                data_dir = path or settings.geosatdb_data_dir

                # Load GEOSatDB
                with console.status(
                    f"[bold blue]Loading GEOSatDB from {data_dir}...",
                    spinner="dots",
                ):
                    counts = await load_geosatdb(session_factory, data_dir)

                # Inform user about the operation status
                console.print(
                    f"[bold green]GEOSatDB loaded[/] - "
                    f"[cyan]{counts['satellites']}[/] satellites, "
                    f"[cyan]{counts['sensors']}[/] sensors, "
                    f"[cyan]{counts['bands']}[/] bands."
                )

            if source in {"spectral-indices", "all"}:
                # Define spectral indices data path
                json_path = path or settings.spectral_indices_path

                # Load spectral indices
                with console.status(
                    f"[bold blue]Loading Spectral Indices from {json_path}...",
                    spinner="dots",
                ):
                    counts = await load_spectral_indices(session_factory, json_path)

                # Inform user about the operation status
                console.print(
                    f"[bold green]Spectral Indices loaded[/] - "
                    f"[cyan]{counts['indices']}[/] indices, "
                    f"[cyan]{counts['band_types']}[/] band types, "
                    f"[cyan]{counts['domains']}[/] domains, "
                    f"[cyan]{counts['platforms']}[/] platforms."
                )

            if source in {"sits-collections", "all"}:
                # Define SITS collections data path
                txt_path = path or settings.sits_collections_path

                # Load SITS collections
                with console.status(
                    f"[bold blue]Loading SITS Collections from {txt_path}...", spinner="dots"
                ):
                    counts = await load_sits_collections(session_factory, txt_path)

                # Inform user about the operation status
                console.print(
                    f"[bold green]SITS Collections loaded[/] - "
                    f"[cyan]{counts['collections']}[/] collections, "
                    f"[cyan]{counts['bands']}[/] bands."
                )

    # Run async function
    asyncio.run(_run())


#
# Create link command
#
@ontology_app.command("link")
def link() -> None:
    """Create cross-ontology links (Platform -> Satellite)."""

    # Define async function
    async def _run() -> None:

        # Run session scope
        async with session_scope() as (_settings, _engine, session_factory):
            # Create cross-ontology links
            with console.status("[bold blue]Linking ontologies...", spinner="dots"):
                # Let's link!
                counts = await link_ontologies(session_factory)

            # Inform user about the operation
            console.print(
                f"[bold green]Done![/] [cyan]{counts['platform_links']}[/] platform links created."
            )

    # Run async function
    asyncio.run(_run())


#
# Create clear command
#
@ontology_app.command("clear")
def clear(
    namespace: str = typer.Option(
        "all",
        "--namespace",
        help="Which ontology to clear: "
        "'geosatdb', 'spectral-indices', 'sits-collections', or 'all'.",
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete ontology rows from vector store."""
    # Define valid namespaces
    valid = {
        "geosatdb",
        "spectral-indices",
        "sits-collections",
        "all",
    }

    # If namespace is invalid, just exit
    if namespace not in valid:
        console.print(
            f"[bold red]Invalid namespace '{namespace}'. Choose from: {', '.join(sorted(valid))}[/]"
        )

        # Exit with error code 1
        raise typer.Exit(code=1)

    # If not confirmed, print confirmation prompt
    if not confirm:
        # Print confirmation prompt
        typer.confirm(
            f"This will delete all '{namespace}' ontology rows. Continue?",
            abort=True,
        )

    # Define async function
    async def _run() -> None:

        # Run session scope
        async with session_scope() as (_settings, _engine, session_factory):
            # Run session
            async with session_factory() as session:
                # If namespace is GEOSatDB or all, clear GEOSatDB
                if namespace in {"geosatdb", "all"}:
                    # Clear GEOSatDB
                    with console.status(
                        "[bold yellow]Clearing GEOSatDB rows...",
                        spinner="dots",
                    ):
                        await session.execute(delete(PlatformSatellite))
                        await session.execute(delete(SensorBand))
                        await session.execute(delete(SensorSatellite))
                        await session.execute(delete(Band))
                        await session.execute(delete(Sensor))
                        await session.execute(delete(Satellite))

                    # Inform user about the operation status
                    console.print("[green]GEOSatDB rows deleted.[/]")

                # If namespace is Spectral Indices or all, clear Spectral Indices
                if namespace in {"spectral-indices", "all"}:
                    # Clear Spectral Indices
                    with console.status(
                        "[bold yellow]Clearing Spectral Indices rows...",
                        spinner="dots",
                    ):
                        await session.execute(delete(IndexBandType))
                        await session.execute(delete(IndexDomain))
                        await session.execute(delete(IndexPlatform))
                        await session.execute(delete(BandType))
                        await session.execute(delete(Domain))
                        await session.execute(delete(Platform))
                        await session.execute(delete(SpectralIndex))

                    console.print("[green]Spectral Indices rows deleted.[/]")

                # If namespace is SITS Collections or all, clear SITS Collections
                if namespace in {"sits-collections", "all"}:
                    # Clear SITS Collections
                    with console.status(
                        "[bold yellow]Clearing SITS Collections rows...",
                        spinner="dots",
                    ):
                        await session.execute(delete(CollectionBand))
                        await session.execute(delete(SitsCollection))

                    # Inform user about the operation status
                    console.print("[green]SITS Collections rows deleted.[/]")

                # Commit session
                await session.commit()

        console.print("[bold green]Ontology cleared.[/]")

    # Run!
    asyncio.run(_run())
