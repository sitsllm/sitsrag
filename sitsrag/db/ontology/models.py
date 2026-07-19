#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Ontology models."""

from sqlalchemy import Boolean, Double, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sitsrag.db.base import Base


class Satellite(Base):
    """Satellite model."""

    __tablename__ = "satellites"

    uri: Mapped[str] = mapped_column(String, primary_key=True)
    """Satellite URI."""

    name: Mapped[str] = mapped_column(String, nullable=False)
    """Name."""

    alternate_name: Mapped[str | None] = mapped_column(String)
    """Alternate name."""

    launch_date: Mapped[str | None] = mapped_column(String)
    """Launch date."""

    owner: Mapped[str | None] = mapped_column(String)
    """Owner."""

    orbit_type: Mapped[str | None] = mapped_column(String)
    """Orbit type."""

    operational_status: Mapped[str | None] = mapped_column(String)
    """Operational status."""

    data_portal: Mapped[str | None] = mapped_column(String)
    """Data portal."""

    sensors: Mapped[list["Sensor"]] = relationship(
        secondary="sensor_satellites",
        back_populates="satellites",
    )
    """Sensors."""


class Sensor(Base):
    """Sensor model."""

    __tablename__ = "sensors"

    uri: Mapped[str] = mapped_column(String, primary_key=True)
    """Sensor URI."""

    name: Mapped[str] = mapped_column(String, nullable=False)
    """Name."""

    alternate_name: Mapped[str | None] = mapped_column(String)
    """Alternate name."""

    sensor_type: Mapped[str | None] = mapped_column(String)
    """Sensor type."""

    max_swath: Mapped[float | None] = mapped_column(Double)
    """Max swath."""

    resolution_best: Mapped[float | None] = mapped_column(Double)
    """Resolution best."""

    revisit_time_best: Mapped[float | None] = mapped_column(Double)
    """Revisit time best."""

    description: Mapped[str | None] = mapped_column(Text)
    """Description."""

    satellites: Mapped[list["Satellite"]] = relationship(
        secondary="sensor_satellites",
        back_populates="sensors",
    )
    """Bands."""

    bands: Mapped[list["Band"]] = relationship(
        secondary="sensor_bands",
        back_populates="sensors",
    )
    """Sensors."""


class SensorSatellite(Base):
    """Sensor satellite model."""

    __tablename__ = "sensor_satellites"

    sensor_uri: Mapped[str] = mapped_column(
        ForeignKey("sensors.uri"),
        primary_key=True,
    )
    """Sensor URI."""

    satellite_uri: Mapped[str] = mapped_column(ForeignKey("satellites.uri"), primary_key=True)
    """Satellite URI."""


class Band(Base):
    """Band model."""

    __tablename__ = "bands"

    uri: Mapped[str] = mapped_column(String, primary_key=True)
    """Band URI."""

    label: Mapped[str | None] = mapped_column(String)
    """Label."""

    waveband_region: Mapped[str | None] = mapped_column(String)
    """Waveband region."""

    bound_min: Mapped[float | None] = mapped_column(Double)
    """Bound min."""

    bound_max: Mapped[float | None] = mapped_column(Double)
    """Bound max."""

    bound_unit: Mapped[str | None] = mapped_column(String, server_default="micrometer")
    """Bound unit."""

    sensors: Mapped[list["Sensor"]] = relationship(secondary="sensor_bands", back_populates="bands")
    """Sensors."""


class SensorBand(Base):
    """Sensor band model."""

    __tablename__ = "sensor_bands"

    sensor_uri: Mapped[str] = mapped_column(ForeignKey("sensors.uri"), primary_key=True)
    """Sensor URI."""

    band_uri: Mapped[str] = mapped_column(ForeignKey("bands.uri"), primary_key=True)
    """Band URI."""


class SpectralIndex(Base):
    """Spectral index model."""

    __tablename__ = "spectral_indices"

    short_name: Mapped[str] = mapped_column(String, primary_key=True)
    """Short name."""

    long_name: Mapped[str | None] = mapped_column(String)
    """Long name."""

    formula: Mapped[str | None] = mapped_column(Text)
    """Formula."""

    application_domain: Mapped[str | None] = mapped_column(String)
    """Application domain."""

    reference: Mapped[str | None] = mapped_column(String)
    """Reference."""

    date_of_addition: Mapped[str | None] = mapped_column(String)
    """Date of addition."""

    band_types: Mapped[list["BandType"]] = relationship(
        secondary="index_band_types", back_populates="indices"
    )
    """Band types."""

    platforms: Mapped[list["Platform"]] = relationship(
        secondary="index_platforms", back_populates="indices"
    )
    """Platforms."""

    domains: Mapped[list["Domain"]] = relationship(
        secondary="index_domains", back_populates="indices"
    )
    """Domains."""


class BandType(Base):
    """Band type model."""

    __tablename__ = "band_types"

    code: Mapped[str] = mapped_column(String, primary_key=True)
    """Band code."""

    indices: Mapped[list["SpectralIndex"]] = relationship(
        secondary="index_band_types", back_populates="band_types"
    )
    """Indices."""


class IndexBandType(Base):
    """Index band type model."""

    __tablename__ = "index_band_types"

    short_name: Mapped[str] = mapped_column(
        ForeignKey("spectral_indices.short_name"),
        primary_key=True,
    )
    """Short name."""

    band_code: Mapped[str] = mapped_column(ForeignKey("band_types.code"), primary_key=True)
    """Band code."""


class Platform(Base):
    """Platform model."""

    __tablename__ = "platforms"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    """Name."""

    indices: Mapped[list["SpectralIndex"]] = relationship(
        secondary="index_platforms",
        back_populates="platforms",
    )
    """Indices."""

    satellites: Mapped[list["Satellite"]] = relationship(secondary="platform_satellites")
    """Satellites."""


class IndexPlatform(Base):
    """Index platform model."""

    __tablename__ = "index_platforms"

    short_name: Mapped[str] = mapped_column(
        ForeignKey("spectral_indices.short_name"),
        primary_key=True,
    )
    """Short name."""

    platform_name: Mapped[str] = mapped_column(ForeignKey("platforms.name"), primary_key=True)
    """Platform name."""


class PlatformSatellite(Base):
    """Platform satellite model."""

    __tablename__ = "platform_satellites"

    platform_name: Mapped[str] = mapped_column(ForeignKey("platforms.name"), primary_key=True)
    """Platform name."""

    satellite_uri: Mapped[str] = mapped_column(ForeignKey("satellites.uri"), primary_key=True)
    """Satellite URI."""


class Domain(Base):
    """Domain model."""

    __tablename__ = "domains"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    """Name."""

    indices: Mapped[list["SpectralIndex"]] = relationship(
        secondary="index_domains",
        back_populates="domains",
    )
    """Indices."""


class IndexDomain(Base):
    """Index domain model."""

    __tablename__ = "index_domains"

    short_name: Mapped[str] = mapped_column(
        ForeignKey("spectral_indices.short_name"),
        primary_key=True,
    )
    """Short name."""

    domain_name: Mapped[str] = mapped_column(ForeignKey("domains.name"), primary_key=True)
    """Domain name."""


class SitsCollection(Base):
    """SITS collection model."""

    __tablename__ = "sits_collections"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    """ID."""

    name: Mapped[str] = mapped_column(String, nullable=False)
    """Name."""

    source: Mapped[str] = mapped_column(String, nullable=False)
    """Source."""

    satellite: Mapped[str | None] = mapped_column(String)
    """Satellite."""

    sensor: Mapped[str | None] = mapped_column(String)
    """Sensor."""

    grid_system: Mapped[str | None] = mapped_column(String)
    """Grid system."""

    period_start: Mapped[str | None] = mapped_column(String)
    """Period start."""

    period_end: Mapped[str | None] = mapped_column(String)
    """Period end."""

    open_data: Mapped[bool] = mapped_column(Boolean, server_default="false")
    """Open data."""

    requires_token: Mapped[bool] = mapped_column(Boolean, server_default="false")
    """Requires token."""

    band_codes: Mapped[list["CollectionBand"]] = relationship(back_populates="collection")
    """Band codes."""


class CollectionBand(Base):
    """Collection band model."""

    __tablename__ = "collection_bands"

    collection_id: Mapped[str] = mapped_column(ForeignKey("sits_collections.id"), primary_key=True)
    """Collection ID."""

    band_code: Mapped[str] = mapped_column(String, nullable=False, primary_key=True)
    """Band code."""

    collection: Mapped["SitsCollection"] = relationship(back_populates="band_codes")
    """Collection."""
