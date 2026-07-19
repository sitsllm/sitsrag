#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Loader for the curated catalog of papers using SITS."""

import json
from pathlib import Path

import pydash as py_
from langchain_core.documents import Document

from sitsrag.config import Settings
from sitsrag.logging import get_logger

#
# Logger
#
logger = get_logger(__name__)


#
# Auxiliary functions
#
def _build_content(record: dict) -> str:
    """Build the text blob that is embedded and full-text indexed.

    Args:
        record (dict): Paper record.

    Returns:
        str: The composed content.
    """
    return "\n".join(
        py_.compact(
            [
                f"Article: {record['title']}",
                f"Authors: {', '.join(record['authors'])}",
                f"Year: {record['year']}",
                f"Venue: {record['venue']}",
                f"Domains: {', '.join(record['domain'])}",
                "",
                record["abstract"],
            ]
        )
    )


#
# High-level interface
#
def parse_articles(path: str | Path | None = None) -> list[Document]:
    """Load the curated articles catalog into document chunks.

    Each article becomes a single ``Document`` (no chunking) so that a catalog
    row stays whole. The composed content is embedded and full-text
    indexed. The citation fields are kept in metadata.

    Args:
        path (str | Path | None): Path to the articles JSON file.

    Returns:
        list[Document]: One document per article.
    """
    # Resolve the path
    articles_path = Settings().articles_path
    articles_path = Path(path) if path is not None else articles_path

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")

    # Load the curated records
    records = json.loads(articles_path.read_text(encoding="utf-8"))

    # Build documents
    documents = [
        Document(
            page_content=_build_content(record),
            metadata={
                "source": "articles",
                "id": record["id"],
                "title": record["title"],
                "authors": record["authors"],
                "year": record["year"],
                "venue": record["venue"],
                "doi": record.get("doi"),
                "url": record["url"],
                "domain": record["domain"],
                "chunk_id": f"article-{record['id']}",
            },
        )
        for record in records
    ]

    logger.info(
        event="Parsed articles",
        path=str(articles_path),
        count=len(documents),
    )

    # Return!
    return documents
