#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Spectral Indices JSON loader."""

import json
from pathlib import Path

import pydash as py_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sitsrag.db.mutators import mut_db_insert_ignore_rows, mut_db_upsert_rows
from sitsrag.db.ontology.models import (
    BandType,
    Domain,
    IndexBandType,
    IndexDomain,
    IndexPlatform,
    Platform,
    SpectralIndex,
)
from sitsrag.logging import get_logger

#
# Logger
#
logger = get_logger(__name__)


#
# High-level interface
#
async def load_spectral_indices(
    session_factory: async_sessionmaker[AsyncSession],
    json_path: Path,
) -> dict[str, int]:
    """Parse spectral indices JSON file and upsert rows into vector store.

    Args:
        session_factory (async_sessionmaker[AsyncSession]): SQLAlchemy async session factory.

        json_path (Path): Path to spectral indices JSON file.

    Returns:
        dict[str, int]: Counts of rows upserted per type.
    """
    # Check if file exists
    if not json_path.exists():
        raise FileNotFoundError(f"Spectral indices file not found: {json_path}")

    # Load JSON file
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    # Get spectral indices
    indices = data.get("SpectralIndices", {})

    # Log loaded spectral indices
    logger.info(
        event="Loaded spectral indices",
        count=len(indices),
        path=str(json_path),
    )

    # Collect unique values and relationship lists
    index_batch = []
    band_rels = []
    domain_rels = []
    platform_rels = []

    # Iterate over spectral indices
    for short_name, entry in indices.items():
        # Add index to batch
        index_batch.append(
            {
                "short_name": short_name,
                "long_name": py_.get(entry, "long_name", ""),
                "formula": py_.get(entry, "formula", ""),
                "application_domain": py_.get(entry, "application_domain", ""),
                "reference": py_.get(entry, "reference", ""),
                "date_of_addition": py_.get(entry, "date_of_addition", ""),
            }
        )

        # Add band relationships to batch
        band_rels.extend(
            py_.map_(
                py_.get(entry, "bands", []), lambda bc: {"short_name": short_name, "band_code": bc}
            )
        )

        # Add domain relationships to batch
        domain = py_.get(entry, "application_domain", "")
        if domain:
            domain_rels.append({"short_name": short_name, "domain_name": domain})

        # Add platform relationships to batch
        platform_rels.extend(
            py_.map_(
                py_.get(entry, "platforms", []),
                lambda p: {"short_name": short_name, "platform_name": p},
            )
        )

    # Get all unique vals
    all_band_codes = py_.uniq(py_.map_(band_rels, "band_code"))
    all_domains = py_.uniq(py_.map_(domain_rels, "domain_name"))
    all_platforms = py_.uniq(py_.map_(platform_rels, "platform_name"))

    # Initialize counts
    counts = {
        "indices": 0,
        "band_types": 0,
        "domains": 0,
        "platforms": 0,
    }

    # Upsert rows into vector store
    async with session_factory() as session:
        # Upsert spectral indices
        counts["indices"] = await mut_db_upsert_rows(
            session=session,
            model=SpectralIndex,
            rows=index_batch,
            key="short_name",
        )

        # Upsert band types
        counts["band_types"] = await mut_db_insert_ignore_rows(
            session=session,
            model=BandType,
            rows=py_.map_(sorted(all_band_codes), lambda c: {"code": c}),
        )

        # Upsert domains
        counts["domains"] = await mut_db_insert_ignore_rows(
            session=session,
            model=Domain,
            rows=py_.map_(sorted(all_domains), lambda n: {"name": n}),
        )

        # Upsert platforms
        counts["platforms"] = await mut_db_insert_ignore_rows(
            session=session,
            model=Platform,
            rows=py_.map_(sorted(all_platforms), lambda n: {"name": n}),
        )

        # Upsert index relationships
        await mut_db_insert_ignore_rows(
            session=session,
            model=IndexBandType,
            rows=band_rels,
        )
        await mut_db_insert_ignore_rows(
            session=session,
            model=IndexDomain,
            rows=domain_rels,
        )
        await mut_db_insert_ignore_rows(
            session=session,
            model=IndexPlatform,
            rows=platform_rels,
        )

        # Commit session
        await session.commit()

    # Log load complete
    logger.info(
        event="Spectral indices load complete",
        indices=counts["indices"],
        band_types=counts["band_types"],
        domains=counts["domains"],
        platforms=counts["platforms"],
    )

    return counts
