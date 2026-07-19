#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""SITS RAG CLI application."""

import pyfiglet
import typer
import uvicorn
from rich.panel import Panel
from rich.text import Text

from sitsrag.cli.db import db_app
from sitsrag.cli.index import index_app
from sitsrag.cli.infra import console
from sitsrag.cli.ontology import ontology_app
from sitsrag.config import Settings
from sitsrag.logging import configure_logging


#
# Utilities
#
def _print_banner() -> None:
    """Print the SITS RAG ASCII banner."""
    # Generate banner text
    banner_text = pyfiglet.figlet_format("SITS RAG", font="slant")

    # Create banner
    banner = Text(banner_text, style="bold cyan")
    subtitle = Text(
        "Satellite Image Time Series  |  AI-Powered Documentation Assistant",
        style="dim",
    )

    # Append subtitle to banner
    banner.append("\n")
    banner.append(subtitle)

    # Print banner
    console.print(Panel(banner, border_style="blue", padding=(0, 2)))


#
# Constants
#

# Verbose levels
VERBOSE_LEVELS = {
    0: "WARNING",
    1: "INFO",
    2: "DEBUG",
}

# Workflow diagram
WORKFLOW_DIAGRAM = """
[bold cyan]SITS RAG Setup Workflow[/]

[dim]Follow these steps to get the system running from scratch:[/]

  [bold white]Step 1[/] [dim]─────[/] [bold blue]Database[/]
  [dim]│[/]         Set up the SQLite schema
  [dim]│[/]         [green]$ sitsrag db migrate[/]
  [dim]│[/]
  [bold white]Step 2[/] [dim]─────[/] [bold blue]Ontology[/]
  [dim]│[/]         Load domain knowledge (satellites, indices, collections)
  [dim]│[/]         [green]$ sitsrag ontology load all[/]
  [dim]│[/]         [green]$ sitsrag ontology link[/]
  [dim]│[/]
  [bold white]Step 3[/] [dim]─────[/] [bold blue]Index Content[/]
  [dim]│[/]         Parse and embed documentation into the vector store
  [dim]│[/]         [green]$ sitsrag index documentation[/]
  [dim]│[/]         [green]$ sitsrag index reference[/]
  [dim]│[/]         [green]$ sitsrag index articles[/]
  [dim]│[/]
  [bold white]Step 4[/] [dim]─────[/] [bold blue]Serve[/]
  [dim]│[/]         Start the RAG API server
  [dim]│[/]         [green]$ sitsrag serve[/]
  [dim]▼[/]
  [bold green]Ready![/] The SITS RAG API is now accepting requests.

[dim]──────────────────────────────────────────────────[/]

[bold yellow]Re-indexing:[/]  To rebuild vectors from scratch:
  [green]$ sitsrag index recreate documentation[/]
  [green]$ sitsrag index recreate reference[/]
  [green]$ sitsrag index recreate articles[/]
"""


#
# Create CLI app
#
app = typer.Typer(
    name="sitsrag",
    help="SITS RAG - AI-powered assistant for the SITS R Package documentation.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Include database commands
app.add_typer(
    typer_instance=db_app,
    name="db",
)

# Include indexing commands
app.add_typer(
    typer_instance=index_app,
    name="index",
)

# Include ontology commands
app.add_typer(
    typer_instance=ontology_app,
    name="ontology",
)


#
# Root commands
#
@app.command()
def workflow() -> None:
    """Show the setup workflow - what to run and in what order."""
    _print_banner()

    # Print workflow diagram
    console.print(
        Panel(
            renderable=WORKFLOW_DIAGRAM,
            border_style="blue",
            title="Getting Started",
        )
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the SITS RAG API server."""
    _print_banner()
    console.print(f"[bold blue]Starting server on[/] [cyan]{host}:{port}[/]")

    # Print auto-reload message
    if reload:
        console.print("[yellow]Auto-reload enabled (development mode)[/]")

    console.print()

    # Run server (importlib wins ;D )
    uvicorn.run(
        "sitsrag.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG). "
        "Default is quiet - only spinners and results are shown.",
    ),
) -> None:
    """SITS RAG - AI-powered assistant for the SITS R Package documentation."""
    # Get verbose level
    level = VERBOSE_LEVELS.get(min(verbose, 2), "WARNING")

    # If command is ``serve``, configure logging with INFO level
    # > We ignore log level for the ``serve`` command because it manages
    # > its own logging (always INFO).
    if ctx.invoked_subcommand == "serve":
        configure_logging(Settings())

    # Otherwise, configure logging with the specified level
    else:
        configure_logging(Settings(), level_override=level)

    # If no subcommand was given, print the banner and help
    if ctx.invoked_subcommand is None:
        # Print banner
        _print_banner()

        # Empty line
        console.print()

        # Help
        ctx.get_help()
