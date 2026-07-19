#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Ontology query-time expansion."""

import pydash as py_
from sqlalchemy import ColumnElement, and_, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sitsrag.db.ontology.models import (
    Satellite,
    Sensor,
    SitsCollection,
    SpectralIndex,
)
from sitsrag.logging import get_logger

#
# Logger
#
logger = get_logger(__name__)


#
# Internal utilities
#
def _keyword_filter(keyword: str, columns: list[ColumnElement]) -> ColumnElement:
    """Build a portable case-insensitive filter.

    Args:
        keyword (str): Search term.

        columns (list[ColumnElement]): List of columns to search.

    Returns:
        ColumnElement: SQLAlchemy filter expression.
    """
    # Split keyword into tokens
    tokens = [t for t in keyword.split() if t] or [keyword]

    # Build filter expression
    return and_(
        *[
            or_(
                *[col.ilike(f"%{tok}%") for col in columns],
            )
            for tok in tokens
        ],
    )


def _relevance_order(keyword: str, primary: ColumnElement) -> list:
    """Order results.

    Args:
        keyword (str): Search term.

        primary (ColumnElement): Primary column to order by.

    Returns:
        list: List of ordering expressions.
    """
    rank = case(
        (func.lower(primary) == keyword.lower(), 0),
        (primary.ilike(f"{keyword}%"), 1),
        else_=2,
    )

    # Return ordering expressions
    return [rank, func.length(primary), primary]


#
# Formatting utilities
#
def format_index_chunk(record: dict) -> str:
    """Format a spectral index record as a compact text chunk.

    Args:
        record (dict): Dict with spectral index record.

    Returns:
        Formatted text chunk.
    """
    short_name = py_.get(record, "short_name", "")
    long_name = py_.get(record, "long_name", "")
    domain = py_.get(record, "application_domain", "")
    formula = py_.get(record, "formula", "")
    bands = py_.get(record, "bands", []) or []

    band_str = py_.join(bands, ", ") if bands else "N/A"

    # Format chunk
    return "\n".join(
        [
            f"[SpectralIndex] {short_name} — {long_name}",
            f"Domain: {domain}",
            f"Formula: {formula}",
            f"Bands: {band_str}",
        ]
    )


def format_satellite_chunk(record: dict) -> str:
    """Format a satellite record as a compact text chunk.

    Args:
        record (dict): Dict with satellite record.

    Returns:
        Formatted text chunk.
    """
    name = py_.get(record, "name", "")
    sensors = py_.get(record, "sensors", [])

    lines = [f"[Satellite] {name}"]

    # Iterate over sensors
    for sen in sensors:
        # Get sensor name and resolution
        sen_name = py_.get(sen, "name", "")
        resolution = py_.get(sen, "resolution_best")
        sensor_line = f"Sensor: {sen_name}"

        # Add resolution if not None
        if resolution is not None:
            sensor_line += f" | Spatial resolution: {resolution}m"

        # Add sensor line to lines
        lines.append(sensor_line)

        # Get band descriptions
        band_descs = py_.compact(
            py_.map_(
                py_.get(sen, "bands", []),
                lambda b: (
                    f"{py_.get(b, 'waveband_region', '')} "
                    f"({py_.get(b, 'bound_min')}–{py_.get(b, 'bound_max')}"
                    f"{py_.get(b, 'bound_unit', 'µm')})"
                    if py_.get(b, "waveband_region") and py_.get(b, "bound_min") is not None
                    else py_.get(b, "waveband_region", "") or None
                ),
            )
        )

        # Add band descriptions if not empty
        if band_descs:
            lines.append(f"  Bands: {py_.join(band_descs, ', ')}")

    # Get computable indices
    indices = py_.get(record, "computable_indices", [])

    # Add computable indices if not empty
    if indices:
        lines.append(f"Related indices: {py_.join(indices, ', ')}")

    # Return formatted chunk
    return "\n".join(lines)


def format_collection_chunk(record: dict) -> str:
    """Format a SITS collection record as a compact text chunk.

    Args:
        record (dict): Dict with collection record.

    Returns:
        Formatted text chunk.
    """
    source = py_.get(record, "source", "")
    name = py_.get(record, "name", "")
    satellite = py_.get(record, "satellite", "")
    sensor = py_.get(record, "sensor", "")
    grid = py_.get(record, "grid_system", "")
    period_start = py_.get(record, "period_start", "")
    period_end = py_.get(record, "period_end", "")
    open_data = py_.get(record, "open_data", False)
    requires_token = py_.get(record, "requires_token", False)
    bands = py_.get(record, "bands", [])

    # Build header
    sat_sensor = f"{satellite}/{sensor}" if sensor else satellite
    header = f"[SITSCollection] {source} / {name} ({sat_sensor})"

    # Build period
    period = (
        f"Period: {period_start} to {period_end}"
        if period_start and period_end and period_start != period_end
        else f"Period: {period_start}"
        if period_start
        else None
    )

    # Build access
    access = (
        "no (restricted)" if not open_data else "yes (requires token)" if requires_token else "yes"
    )

    # Build meta parts
    meta_parts = py_.compact(
        [
            grid and f"Grid: {grid}",
            period,
            f"Open data: {access}",
        ]
    )

    # Return formatted chunk
    return "\n".join(
        py_.compact(
            [
                header,
                py_.join(meta_parts, " | ") if meta_parts else None,
                bands and f"Bands: {py_.join(bands, ' ')}",
            ]
        )
    )


#
# Ontology search utilities
#
async def search_indices(
    session: AsyncSession,
    keyword: str,
) -> list[dict]:
    """Search spectral indices.

    Args:
        session (AsyncSession): Active async database session.

        keyword (str): Search term (e.g. ``"NDVI"``, ``"vegetation"``).

    Returns:
        list[dict]: List of dicts suitable for ``format_index_chunk()``.
    """
    # Build statement
    stmt = (
        select(SpectralIndex)
        .options(selectinload(SpectralIndex.band_types))
        .where(
            _keyword_filter(
                keyword,
                [
                    SpectralIndex.short_name,
                    SpectralIndex.long_name,
                    SpectralIndex.application_domain,
                ],
            )
        )
        .order_by(*_relevance_order(keyword, SpectralIndex.short_name))
        .limit(10)
    )

    # Execute
    result = await session.execute(stmt)
    indices = result.scalars().unique().all()

    # Build results
    return [
        {
            "short_name": idx.short_name,
            "long_name": idx.long_name,
            "formula": idx.formula,
            "application_domain": idx.application_domain,
            "bands": [bt.code for bt in idx.band_types],
        }
        for idx in indices
    ]


async def search_satellites(
    session: AsyncSession,
    keyword: str,
) -> list[dict]:
    """Search satellites.

    Args:
        session (AsyncSession): Active async database session.

        keyword (str): Search term (e.g. ``"sentinel"``, ``"landsat"``).

    Returns:
        list[dict]: List of dicts suitable for ``format_satellite_chunk()``.
    """
    stmt = (
        select(Satellite)
        .options(selectinload(Satellite.sensors).selectinload(Sensor.bands))
        .where(_keyword_filter(keyword, [Satellite.name, Satellite.alternate_name]))
        .order_by(*_relevance_order(keyword, Satellite.name))
        .limit(5)
    )

    # Execute
    result = await session.execute(stmt)
    sats = result.scalars().unique().all()

    # Build results
    records = []

    # Iterate over satellites
    for sat in sats:
        # Build sensors
        sensors_data = []

        # Iterate over sensors
        for sen in sat.sensors:
            # Build bands
            bands_data = [
                {
                    "waveband_region": b.waveband_region,
                    "bound_min": b.bound_min,
                    "bound_max": b.bound_max,
                    "bound_unit": b.bound_unit,
                }
                for b in sen.bands
            ]

            # Add sensor data
            sensors_data.append(
                {
                    "name": sen.name,
                    "resolution_best": sen.resolution_best,
                    "bands": bands_data,
                }
            )

        # Add satellite data
        records.append(
            {
                "uri": sat.uri,
                "name": sat.name,
                "sensors": sensors_data,
                "computable_indices": [],  # populated by agent if needed
            }
        )

    # Return!
    return records


async def search_collections(
    session: AsyncSession,
    keyword: str,
) -> list[dict]:
    """Search collections available in SITS.

    Args:
        session (AsyncSession): Active async database session.

        keyword (str): Search term (e.g. ``"sentinel"``, ``"BDC"``).

    Returns:
        list[dict]: List of dicts suitable for ``format_collection_chunk()``.
    """
    stmt = (
        select(SitsCollection)
        .options(selectinload(SitsCollection.band_codes))
        .where(
            _keyword_filter(
                keyword,
                [
                    SitsCollection.name,
                    SitsCollection.source,
                    SitsCollection.satellite,
                    SitsCollection.sensor,
                ],
            )
        )
        .order_by(*_relevance_order(keyword, SitsCollection.name))
        .limit(10)
    )

    # Execute
    result = await session.execute(stmt)
    collections = result.scalars().unique().all()

    # Build results
    return [
        {
            "id": col.id,
            "name": col.name,
            "source": col.source,
            "satellite": col.satellite,
            "sensor": col.sensor,
            "grid_system": col.grid_system,
            "period_start": col.period_start,
            "period_end": col.period_end,
            "open_data": col.open_data,
            "requires_token": col.requires_token,
            "bands": [cb.band_code for cb in col.band_codes],
        }
        for col in collections
    ]
