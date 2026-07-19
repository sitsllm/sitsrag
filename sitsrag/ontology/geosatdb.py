#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""GEOSatDB RDF loader."""

from pathlib import Path

import pydash as py_
from rdflib import RDF, Graph, Namespace
from rdflib.namespace import RDFS
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sitsrag.db.mutators import mut_db_insert_ignore_rows, mut_db_upsert_rows
from sitsrag.db.ontology.models import Band, Satellite, Sensor, SensorBand, SensorSatellite
from sitsrag.logging import get_logger
from sitsrag.mutators.rdf import (
    mut_build_label_map,
    mut_extract_uri_tail,
    mut_graph_field_float,
    mut_graph_field_str,
    mut_graph_field_str_list,
    mut_resolve_uri,
)

#
# Logger
#
logger = get_logger(__name__)

#
# Constants
#

# RDF Namespaces
EO_ONT = Namespace("https://www.eoknowledgehub.cn/eo/ontology/")
EOR = Namespace("https://www.eoknowledgehub.cn/eo/resource/")
SCHEMA = Namespace("http://schema.org/")
MAC = Namespace("https://schemas.isotc211.org/19115/-2/mac/2.2/")
MRC = Namespace("https://schemas.isotc211.org/19115/-1/mrc/1.3.0/")

# SITS-relevant platform patterns
SITS_PLATFORM_PATTERNS = [
    "sentinel",
    "landsat",
    "modis",
    "terra",
    "aqua",
    "cbers",
    "planet",
    "planetscope",
    "spot",
    "alos",
]


#
# Utilities
#
def _is_sits_relevant(name: str, alt_names: list[str]) -> bool:
    """Check if the satellite name matches any SITS platform pattern."""
    combined = py_.join([name, *alt_names], " ").lower()

    return py_.some(
        collection=SITS_PLATFORM_PATTERNS,
        predicate=lambda p: p in combined,
    )


#
# Parsing
#
def _parse_graphs(data_dir: Path) -> tuple[Graph, Graph, Graph]:
    """Load and validate GEOSatDB TTL files.

    Args:
        data_dir (Path): Directory containing GEOSatDB TTL files.

    Returns:
        tuple[Graph, Graph, Graph]: Satellite graph, sensor graph, and sensor2satellite graph.
    """
    # Get paths
    satellite_ttl = data_dir / "satellite.ttl"
    sensor_ttl = data_dir / "sensor.ttl"
    s2s_ttl = data_dir / "sensor2satellite.ttl"

    # Check if paths exist
    for path in (satellite_ttl, sensor_ttl, s2s_ttl):
        if not path.exists():
            raise FileNotFoundError(f"GEOSatDB file not found: {path}")

    # Parse graphs
    sat_graph = Graph()
    sat_graph.parse(str(satellite_ttl), format="turtle")

    # Parse sensor graph
    sen_graph = Graph()
    sen_graph.parse(str(sensor_ttl), format="turtle")

    # Parse sensor2satellite graph
    s2s_graph = Graph()
    s2s_graph.parse(str(s2s_ttl), format="turtle")

    # Return graphs
    return sat_graph, sen_graph, s2s_graph


#
# Extraction
#
def _extract_satellites(
    sat_graph: Graph,
    owner_labels: dict[str, str],
    status_labels: dict[str, str],
    orbit_labels: dict[str, str],
) -> tuple[list[dict], set[str]]:
    """Extract SITS-relevant satellites.

    Args:
        sat_graph (Graph): Satellite graph.

        owner_labels (dict[str, str]): Owner labels.

        status_labels (dict[str, str]): Status labels.

        orbit_labels (dict[str, str]): Orbit labels.

    Returns:
        tuple[list[dict], set[str]]: Satellite rows and relevant URI set.
    """
    sat_batch = []
    sat_uris = set()

    # Iterate over satellites
    for sat_uri in sat_graph.subjects(RDF.type, EO_ONT.Satellite):
        # Get fields
        name = mut_graph_field_str(
            graph=sat_graph,
            subject=sat_uri,
            predicate=SCHEMA.name,
            default="",
        )

        alt_names = mut_graph_field_str_list(
            graph=sat_graph,
            subject=sat_uri,
            predicate=SCHEMA.alternateName,
            default=[],
        )

        # If not relevant, skip it
        if not _is_sits_relevant(name, alt_names):
            continue

        # Add satellite URI to set
        sat_uris.add(str(sat_uri))
        sat_batch.append(
            {
                "uri": str(sat_uri),
                "name": name,
                "alternate_name": ", ".join(alt_names),
                "launch_date": mut_graph_field_str(
                    graph=sat_graph,
                    subject=sat_uri,
                    predicate=EO_ONT.launchDate,
                    default="",
                ),
                "owner": mut_resolve_uri(
                    graph=sat_graph,
                    subject=sat_uri,
                    predicate=EO_ONT.owner,
                    labels=owner_labels,
                ),
                "orbit_type": mut_resolve_uri(
                    graph=sat_graph,
                    subject=sat_uri,
                    predicate=EO_ONT.orbitType,
                    labels=orbit_labels,
                ),
                "operational_status": mut_resolve_uri(
                    graph=sat_graph,
                    subject=sat_uri,
                    predicate=EO_ONT.operationalStatus,
                    labels=status_labels,
                ),
                "data_portal": mut_graph_field_str(
                    graph=sat_graph,
                    subject=sat_uri,
                    predicate=EO_ONT.dataPortal,
                    default="",
                ),
            }
        )

    # Return batches and URIs
    return sat_batch, sat_uris


def _extract_sensors(
    sen_graph: Graph,
    s2s_graph: Graph,
    sat_uris: set[str],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Extract SITS-relevant sensors.

    Args:
        sen_graph (Graph): Sensor graph.

        s2s_graph (Graph): Sensor2satellite graph.

        sat_uris (set[str]): Set of SITS-relevant satellite URIs.

    Returns:
        tuple[list[dict], list[dict], list[dict], list[dict]]: Sensor tuple.
    """
    # Build sensor -> satellite mapping
    sensor_to_sats = {}

    # easy way passing =)
    for sen_uri, _, sat_uri in s2s_graph.triples((None, MAC.mountedOn, None)):
        # If relevant, skip it
        if str(sat_uri) in sat_uris:
            sensor_to_sats.setdefault(str(sen_uri), []).append(str(sat_uri))

    # Build relevant sensor URIs
    relevant_sensor_uris = set(sensor_to_sats.keys())

    # Initialize batches
    sen_batch = []
    band_batch = []
    sensor_band_rels = []
    sensor_sat_rels = []

    # Iterate over sensors
    for sen_uri in sen_graph.subjects(RDF.type, EO_ONT.Sensor):
        if str(sen_uri) not in relevant_sensor_uris:
            continue

        # Get fields
        name = mut_graph_field_str(
            graph=sen_graph,
            subject=sen_uri,
            predicate=SCHEMA.name,
            default="",
        )
        alt_names = mut_graph_field_str_list(
            graph=sen_graph,
            subject=sen_uri,
            predicate=SCHEMA.alternateName,
            default=[],
        )
        description = mut_graph_field_str(
            graph=sen_graph,
            subject=sen_uri,
            predicate=SCHEMA.description,
            default="",
        )

        # Get sensor type
        st_uri_obj = py_.head(list(sen_graph.objects(sen_uri, EO_ONT.sensorType)))
        sensor_type = mut_extract_uri_tail(str(st_uri_obj)) if st_uri_obj else ""

        # Add sensor data to batch
        sen_batch.append(
            {
                "uri": str(sen_uri),
                "name": name,
                "alternate_name": ", ".join(alt_names),
                "sensor_type": sensor_type,
                "max_swath": mut_graph_field_float(
                    graph=sen_graph,
                    subject=sen_uri,
                    predicate=EO_ONT.maxSwath,
                ),
                "resolution_best": mut_graph_field_float(
                    graph=sen_graph,
                    subject=sen_uri,
                    predicate=EO_ONT.resolutionBest,
                ),
                "revisit_time_best": mut_graph_field_float(
                    graph=sen_graph,
                    subject=sen_uri,
                    predicate=EO_ONT.revisitTimeBest,
                ),
                "description": description[:500] if description else "",
            }
        )

        # Add sensor -> satellite relationships
        sensor_sat_rels.extend(
            [
                {"sensor_uri": str(sen_uri), "satellite_uri": sat_uri_str}
                for sat_uri_str in sensor_to_sats.get(str(sen_uri), [])
            ]
        )

        # Iterate over bands
        for band_uri in sen_graph.objects(sen_uri, EO_ONT.operationalBand):
            waveband_regions = py_.map_(
                list(sen_graph.objects(band_uri, EO_ONT.wavebandRegion)),
                lambda uri: mut_extract_uri_tail(str(uri)),
            )

            # Add band data
            band_batch.append(
                {
                    "uri": str(band_uri),
                    "label": mut_graph_field_str(
                        graph=sen_graph,
                        subject=band_uri,
                        predicate=RDFS.label,
                        default="",
                    ),
                    "waveband_region": ", ".join(waveband_regions),
                    "bound_min": mut_graph_field_float(
                        graph=sen_graph,
                        subject=band_uri,
                        predicate=MRC.boundMin,
                    ),
                    "bound_max": mut_graph_field_float(
                        graph=sen_graph,
                        subject=band_uri,
                        predicate=MRC.boundMax,
                    ),
                    "bound_unit": (
                        mut_graph_field_str(
                            graph=sen_graph,
                            subject=band_uri,
                            predicate=MRC.boundUnit,
                            default="micrometer",
                        )
                    ),
                }
            )

            # Add sensor -> band relationships
            sensor_band_rels.append(
                {
                    "sensor_uri": str(sen_uri),
                    "band_uri": str(band_uri),
                }
            )

    # Return!
    return sen_batch, band_batch, sensor_sat_rels, sensor_band_rels


#
# High-level interface
#
async def load_geosatdb(
    session_factory: async_sessionmaker[AsyncSession],
    data_dir: Path,
) -> dict[str, int]:
    """Parse GEOSatDB TTL files and upsert satellite/sensor/band rows.

    Args:
        session_factory (async_sessionmaker[AsyncSession]): SQLAlchemy async session factory.

        data_dir (Path): Directory containing GEOSatDB TTL files.

    Returns:
        dict[str, int]: Counts of rows upserted.
    """
    # Parse graphs
    logger.info("Parsing GEOSatDB TTL files", data_dir=str(data_dir))
    sat_graph, sen_graph, s2s_graph = _parse_graphs(data_dir)

    # Build label
    owner_labels = mut_build_label_map(
        graph=sat_graph,
        rdf_type=EO_ONT.Owner,
        value_predicate=EO_ONT.ownerId,
        fallback_predicate=RDFS.label,
    )

    # Build status
    status_labels = mut_build_label_map(
        graph=sat_graph,
        rdf_type=EO_ONT.OperationalStatus,
        value_predicate=EO_ONT.operationalStatusId,
    )

    # Build orbit
    orbit_labels = mut_build_label_map(
        graph=sat_graph,
        rdf_type=EO_ONT.OrbitType,
        value_predicate=RDFS.label,
    )

    # Extract satellites
    sat_batch, sat_uris = _extract_satellites(
        sat_graph=sat_graph,
        owner_labels=owner_labels,
        status_labels=status_labels,
        orbit_labels=orbit_labels,
    )

    # Log found satellites
    logger.info("Found SITS-relevant satellites", count=len(sat_batch))

    # Extract sensors
    sen_batch, band_batch, sensor_sat_rels, sensor_band_rels = _extract_sensors(
        sen_graph=sen_graph,
        s2s_graph=s2s_graph,
        sat_uris=sat_uris,
    )

    # Log found sensors and bands
    logger.info(
        "Found sensors and bands for SITS-relevant satellites",
        sensors=len(sen_batch),
        bands=len(band_batch),
    )

    # Upsert rows into PostgreSQL
    async with session_factory() as session:
        # Upsert satellites
        counts = {
            "satellites": await mut_db_upsert_rows(
                session=session,
                model=Satellite,
                rows=sat_batch,
                key="uri",
            ),
            "sensors": await mut_db_upsert_rows(
                session=session,
                model=Sensor,
                rows=sen_batch,
                key="uri",
            ),
            "bands": await mut_db_upsert_rows(
                session=session,
                model=Band,
                rows=band_batch,
                key="uri",
            ),
        }

        # Upsert sensor -> satellite relationships
        await mut_db_insert_ignore_rows(
            session=session,
            model=SensorSatellite,
            rows=sensor_sat_rels,
        )

        # Upsert sensor -> band relationships
        await mut_db_insert_ignore_rows(
            session=session,
            model=SensorBand,
            rows=sensor_band_rels,
        )

        # Commit session
        await session.commit()

    # Log load complete
    logger.info(
        "GEOSatDB load complete",
        satellites=counts["satellites"],
        sensors=counts["sensors"],
        bands=counts["bands"],
    )

    return counts
