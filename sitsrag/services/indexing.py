#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Index service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sitsrag.logging import get_logger
from sitsrag.parsers.html_parser import parse_documentation
from sitsrag.parsers.papers_parser import parse_articles
from sitsrag.parsers.reference_parser import parse_reference

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from sitsrag.config import Settings
    from sitsrag.db.graph.vector_store import SqliteVecStore

#
# Logger
#
logger = get_logger(__name__)


#
# High-level interface
#
class IndexingService:
    """Orchestrates the indexing pipeline for all content sources."""

    def __init__(self, vector_store: SqliteVecStore, settings: Settings) -> None:
        """Initializer.

        Args:
            vector_store (SqliteVecStore): Vector store.

            settings (Settings): Configuration.
        """
        self._store = vector_store
        self._settings = settings

    async def index_documentation(self) -> int:
        """Parse and index the SITS book documentation.

        Returns:
            int: The number of documents stored.
        """
        logger.info("Starting documentation indexing...")

        # Parse documentation
        documents = await parse_documentation(self._settings.sitsbook_url)

        # Store documents
        return await self._store_documents(documents)

    async def index_reference(self) -> int:
        """Parse and index the SITS function reference."""
        logger.info("Starting reference indexing...")

        # Parse reference
        documents = await parse_reference(self._settings.sits_reference_url)

        # Store documents
        return await self._store_documents(documents)

    async def index_articles(self, path: str | None = None) -> int:
        """Parse and index the curated catalog of articles using SITS."""
        logger.info("Starting articles indexing", path=path)

        # Parse the curated article catalog
        documents = parse_articles(path)

        # Store documents
        return await self._store_documents(documents)

    async def reindex_documentation(self) -> int:
        """Delete existing documentation chunks and re-index.

        Returns:
            int: The number of documents stored.
        """
        logger.info("Deleting existing documentation vectors...")

        # Delete existing documentation vectors
        await self._store.adelete(filter={"source": "documentation"})

        # Index documentation
        return await self.index_documentation()

    async def reindex_reference(self) -> int:
        """Delete existing reference chunks and re-index."""
        logger.info("Deleting existing reference vectors...")

        # Delete existing reference vectors
        await self._store.adelete(filter={"source": "reference"})
        await self._store.adelete(filter={"source": "reference_full"})

        # Index reference
        return await self.index_reference()

    async def reindex_articles(self, path: str | None = None) -> int:
        """Delete existing article chunks and re-index.

        Args:
            path (str | None): Optional path to the articles JSON file.

        Returns:
            int: The number of documents stored.
        """
        logger.info("Deleting existing article vectors...")

        # Delete existing article vectors
        await self._store.adelete(filter={"source": "articles"})

        # Index articles
        return await self.index_articles(path)

    async def _store_documents(self, documents: list[Document]) -> int:
        """Store documents in the SQLite-vec vector store.

        Args:
            documents (list[Document]): The list of documents to store.

        Returns:
            int: The number of documents stored.
        """
        if not documents:
            logger.warning("No documents to index")
            return 0

        # Index documents
        logger.info("Indexing documents", count=len(documents))
        await self._store.aadd_documents(documents)

        # Log success
        logger.info("Indexed documents", count=len(documents))

        # Return!
        return len(documents)
