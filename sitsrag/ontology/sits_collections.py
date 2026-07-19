#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""SITS Collections loader."""

import re
from pathlib import Path

import pydash as py_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sitsrag.db.mutators import mut_db_insert_ignore_rows, mut_db_upsert_rows
from sitsrag.db.ontology.models import CollectionBand, SitsCollection
from sitsrag.logging import get_logger

#
# Logger
#
logger = get_logger(__name__)


#
# Utilities
#
def _parse_metadata(text: str) -> list[dict]:
    """Parse SITS collections file into a list of collection dicts.

    Args:
        text (str): Raw content of SITS collections file.

    Returns:
        list[dict]: Parsed collection entries.
    """
    collections = []
    entry_lines = []
    current_source = ""

    def _flush(src: str, lines: list[str]) -> dict | None:
        """Convert buffered lines into a collection dict."""
        if not lines:
            return None

        # Match header (i.e. collection name and satellite/sensor)
        header_match = re.match(r"^-\s+([\w\-\.]+)\s+\(([^)]+)\)\s*$", lines[0].strip())

        # Check if header matches
        if not header_match:
            return None

        # Extract collection name and satellite/sensor
        coll_name = header_match.group(1).strip()
        sat_sensor = header_match.group(2).strip()

        # Extract satellite and sensor
        if "/" in sat_sensor:
            satellite, sensor = sat_sensor.split("/", 1)
        else:
            satellite, sensor = sat_sensor, ""

        # Initialize variables
        grid_system = ""
        period_start = ""
        period_end = None
        bands = []
        open_data = False
        requires_token = False

        # Iterate over properties
        for raw_prop in lines[1:]:
            prop = raw_prop.strip().lstrip("- ").strip()

            # Extract grid system
            if prop.startswith("grid system:"):
                grid_system = prop[len("grid system:") :].strip()

            # Extract period
            elif prop.startswith("period:"):
                raw = prop[len("period:") :].strip()

                # Check if period has start and end
                if "to" in raw:
                    parts = raw.split("to", 1)
                    period_start = parts[0].strip()
                    period_end = parts[1].strip()

                # Check if period has only start
                elif raw:
                    period_start = raw
                    period_end = raw

            # Extract bands
            elif prop.startswith("bands:"):
                band_str = prop[len("bands:") :].strip()
                bands = band_str.split() if band_str else []

            # Extract open data and requires token
            elif "opendata collection" in prop or "not opendata" in prop:
                open_data = "not opendata" not in prop
                requires_token = "requires access token" in prop

        # Return collection dict
        return {
            "id": f"{src}:{coll_name}",
            "source": src,
            "name": coll_name,
            "satellite": satellite,
            "sensor": sensor,
            "grid_system": grid_system,
            "period_start": period_start,
            "period_end": period_end or "",
            "open_data": open_data,
            "requires_token": requires_token,
            "bands": bands,
        }

    def _maybe_flush() -> None:
        """Flush buffered lines if any and current source is set."""
        # non local =)
        nonlocal entry_lines

        # Check if there are lines to flush and current source is set
        if entry_lines and current_source:
            rec = _flush(current_source, entry_lines)

            # Check if record is valid
            if rec:
                collections.append(rec)

        # Clear entry lines
        entry_lines = []

    # Iterate over lines
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        source_match = re.match(r"^([A-Z][A-Z0-9\-]+):$", line.strip())

        # Check if source matches
        if source_match:
            _maybe_flush()
            current_source = source_match.group(1)

        # Check if line is empty
        elif not line.strip():
            _maybe_flush()

        # Check if current source is set
        elif current_source:
            # Add line to entry lines
            entry_lines.append(line)

    # Flush buffered lines
    _maybe_flush()

    # Return collections
    return collections


#
# High-level interface
#
async def load_sits_collections(
    session_factory: async_sessionmaker[AsyncSession],
    txt_path: Path,
) -> dict[str, int]:
    """Parse SITS collections file and upsert rows into PostgreSQL.

    Args:
        session_factory (async_sessionmaker[AsyncSession]): SQLAlchemy async session factory.

        txt_path (Path): Path to SITS collections file.

    Returns:
        dict[str, int]: Counts of rows upserted per type.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"SITS collections file not found: {txt_path}")

    # Read text
    text = txt_path.read_text(encoding="utf-8")

    # Parse metadata
    collections = _parse_metadata(text)

    # Log
    logger.info(
        "Parsed SITS collection entries",
        count=len(collections),
        path=str(txt_path),
    )

    # Initialize counts
    counts = {"collections": 0, "bands": 0}

    # Upsert collections
    async with session_factory() as session:
        # Build rows
        collection_rows = [py_.omit(c, "bands") for c in collections]
        counts["collections"] = await mut_db_upsert_rows(
            session=session,
            model=SitsCollection,
            rows=collection_rows,
            key="id",
        )

        # Build bands rows
        band_rows = py_.flat_map(
            collection=collections,
            iteratee=lambda c: py_.map_(
                c["bands"],
                lambda b: {"collection_id": c["id"], "band_code": b},
            ),
        )

        counts["bands"] = await mut_db_insert_ignore_rows(
            session=session,
            model=CollectionBand,
            rows=band_rows,
        )

        await session.commit()

    logger.info(
        "SITS collections load complete",
        collections=counts["collections"],
        bands=counts["bands"],
    )

    return counts
