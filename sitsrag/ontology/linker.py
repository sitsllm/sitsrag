#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Cross-ontology linker."""

import pydash as py_
from sqlalchemy import or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sitsrag.db.ontology.models import Platform, PlatformSatellite, Satellite
from sitsrag.logging import get_logger

#
# Logger
#
logger = get_logger(__name__)

#
# Constants
#
PLATFORM_SAT_KEYWORDS = {
    "Sentinel-2": ["sentinel-2", "sentinel 2"],
    "Sentinel-1": ["sentinel-1", "sentinel 1"],
    "Landsat-OLI": ["landsat-8", "landsat-9", "landsat 8", "landsat 9"],
    "Landsat-TM": ["landsat-4", "landsat-5", "landsat 4", "landsat 5"],
    "Landsat-ETM+": ["landsat-7", "landsat 7"],
    "MODIS": ["terra", "aqua", "modis"],
    "Planet": ["planet", "planetscope"],
}


#
# High-level interface
#
async def link_ontologies(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, int]:
    """Create cross-ontology links between platforms and satellites.

    For each entry in ``PLATFORM_SAT_KEYWORDS``, finds matching satellites
    by case-insensitive substring match on ``name`` or ``alternate_name``,
    then inserts rows into the ``platform_satellites`` junction table
    with ``ON CONFLICT DO NOTHING``.

    Args:
        session_factory (async_sessionmaker[AsyncSession]): SQLAlchemy async session factory.

    Returns:
        dict[str, int]: Counts of platform-satellite links created.
    """
    counts = {"platform_links": 0}

    async with session_factory() as session:
        # Iterate over platforms
        for platform_name, keywords in PLATFORM_SAT_KEYWORDS.items():
            # Check if the platform exists
            platform = await session.get(Platform, platform_name)

            if not platform:
                logger.debug("Platform not found, skipping", platform=platform_name)
                continue

            # Find matching satellites by keyword
            conditions = py_.flat_map(
                keywords,
                lambda kw: [
                    Satellite.name.ilike(f"%{kw}%"),
                    Satellite.alternate_name.ilike(f"%{kw}%"),
                ],
            )

            # Build statement
            stmt = select(Satellite.uri).where(or_(*conditions))

            # Execute
            result = await session.execute(stmt)
            sat_uris = py_.map_(result.all(), lambda row: row[0])

            # Build links
            for sat_uri in sat_uris:
                link_stmt = (
                    sqlite_insert(PlatformSatellite)
                    .values(platform_name=platform_name, satellite_uri=sat_uri)
                    .on_conflict_do_nothing()
                )

                # Execute statement
                await session.execute(link_stmt)

            # Update counts
            platform_link_size = len(sat_uris)
            counts["platform_links"] += platform_link_size

            # Log if any satellites were linked
            if platform_link_size:
                logger.debug(
                    "Platform linked to satellites",
                    platform=platform_name,
                    count=platform_link_size,
                )

        # Commit session
        await session.commit()

    # Log complete
    logger.info(
        "Cross-ontology linking complete",
        platform_links=counts["platform_links"],
    )

    return counts
