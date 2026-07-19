#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the curated article catalog loader."""

import json
from pathlib import Path

from sitsrag.parsers.papers_parser import parse_articles

#
# Minimal valid record
#
VALID_RECORD = {
    "id": "picoli2018",
    "title": "Big Earth Observation Time Series Analysis for Brazilian Agriculture",
    "authors": ["Michelle Picoli", "Gilberto Camara"],
    "year": 2018,
    "venue": "ISPRS Journal of Photogrammetry and Remote Sensing",
    "doi": "10.1016/j.isprsjprs.2018.08.007",
    "url": "https://doi.org/10.1016/j.isprsjprs.2018.08.007",
    "domain": ["agriculture", "land-use"],
    "abstract": "We analyse satellite image time series to monitor agriculture.",
}


def _write(tmp_path: Path, records: list[dict]) -> Path:
    """Write records to a temporary articles JSON file."""
    path = tmp_path / "articles.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    # Return path
    return path


def test_parse_articles_builds_one_document_per_article(tmp_path: Path):
    """A valid record yields exactly one Document with article metadata."""
    # Write records
    path = _write(tmp_path, [VALID_RECORD])

    # Parse articles
    docs = parse_articles(path)

    # Assert result
    assert len(docs) == 1
    meta = docs[0].metadata

    # Assert metadata is preserved
    assert meta["source"] == "articles"
    assert meta["chunk_id"] == "article-picoli2018"
    assert meta["domain"] == ["agriculture", "land-use"]
    assert meta["authors"] == ["Michelle Picoli", "Gilberto Camara"]
    assert meta["doi"] == "10.1016/j.isprsjprs.2018.08.007"

    # Assert the composed blob carries domain tags and the abstract
    content = docs[0].page_content

    assert "Domains: agriculture, land-use" in content
    assert "monitor agriculture" in content
