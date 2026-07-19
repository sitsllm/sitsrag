#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Vectore store module."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import TYPE_CHECKING

import sqlite_vec
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from sqlalchemy import text

from sitsrag.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from sitsrag.config import Settings


#
# Logger
#
logger = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_embedder(model_name: str, threads: int | None) -> FastEmbedEmbeddings:
    """Load (and cache) a fastembed embedder via the langchain primitive."""
    logger.info(
        event="embedder.loading",
        model=model_name,
        threads=threads,
    )

    return FastEmbedEmbeddings(model_name=model_name, threads=threads)


class SqliteVecStore:
    """Vector + FTS store over a single SQLite file."""

    def __init__(self, engine: AsyncEngine, settings: Settings) -> None:
        """Initializer.

        Args:
            engine (AsyncEngine): SQLAlchemy engine.

            settings (Settings): Application settings.
        """
        self._engine = engine
        self._settings = settings
        self._model_name = settings.embedding_model
        self._dim = settings.embedding_dim
        self._threads = settings.inference_threads

    #
    # Embeddings
    #
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return _load_embedder(self._model_name, self._threads).embed_query(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return _load_embedder(self._model_name, self._threads).embed_documents(texts)

    #
    # Schema
    #
    async def ensure_schema(self) -> None:
        """Create store schema if it does not exist."""
        async with self._engine.begin() as conn:
            # Chunks table
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS chunks ("
                    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    " content TEXT NOT NULL,"
                    " metadata TEXT NOT NULL DEFAULT '{}',"
                    " source TEXT)"
                )
            )

            # Vector chunks table
            await conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                    " chunk_id INTEGER PRIMARY KEY,"
                    " source TEXT PARTITION KEY,"
                    f" embedding FLOAT[{self._dim}])"
                )
            )

            # FTS chunks table
            await conn.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks "
                    "USING fts5(content, content='chunks', content_rowid='id')"
                )
            )

    #
    # Index
    #
    async def aadd_documents(self, documents: list[Document]) -> None:
        """Embed and insert documents into store."""
        if not documents:
            return

        await self.ensure_schema()

        # Get content
        contents = [d.page_content for d in documents]

        # Embed content
        vectors = self.embed_documents(contents)

        # Insert documents into store
        async with self._engine.begin() as conn:
            # Iterate documents
            for doc, vec in zip(documents, vectors, strict=True):
                # Get source
                source = doc.metadata.get("source")

                # Insert document into chunks table
                result = await conn.execute(
                    text(
                        "INSERT INTO chunks (content, metadata, source) "
                        "VALUES (:content, :metadata, :source)"
                    ),
                    {
                        "content": doc.page_content,
                        "metadata": json.dumps(doc.metadata),
                        "source": source,
                    },
                )

                # Get chunk id
                chunk_id = result.lastrowid

                # Insert document into vector chunks table
                await conn.execute(
                    text(
                        "INSERT INTO vec_chunks (chunk_id, source, embedding) "
                        "VALUES (:id, :source, :embedding)"
                    ),
                    {
                        "id": chunk_id,
                        "source": source,
                        "embedding": sqlite_vec.serialize_float32(vec),
                    },
                )

                # Insert document into FTS chunks table
                await conn.execute(
                    text("INSERT INTO fts_chunks (rowid, content) VALUES (:id, :content)"),
                    {"id": chunk_id, "content": doc.page_content},
                )

        # Log results
        logger.info(
            event="vector_store.added",
            count=len(documents),
        )

    async def adelete(self, filter: dict[str, str]) -> None:  # noqa: A002
        """Delete chunks matching an equality filter on source."""
        source = filter.get("source")

        if source is None:
            return

        # Delete chunks from store
        async with self._engine.begin() as conn:
            # Get chunks
            rows = (
                await conn.execute(
                    text("SELECT id FROM chunks WHERE source = :source"),
                    {"source": source},
                )
            ).fetchall()

            # Get ids
            ids = [r[0] for r in rows]

            if not ids:
                return

            # Get ids list
            id_list = ",".join(str(i) for i in ids)

            # Delete from tables
            await conn.execute(text(f"DELETE FROM vec_chunks WHERE chunk_id IN ({id_list})"))
            await conn.execute(text(f"DELETE FROM fts_chunks WHERE rowid IN ({id_list})"))
            await conn.execute(text(f"DELETE FROM chunks WHERE id IN ({id_list})"))

        # Log results
        logger.info(
            event="vector_store.deleted",
            source=source,
            count=len(ids),
        )

    #
    # Retrieval
    #
    async def vector_search(
        self,
        query: str,
        k: int,
        source: str | None = None,
    ) -> list[tuple[int, Document, float]]:
        """K-nearest-neighbour search

        Args:
            query (str): Query string.

            k (int): Number of results to return.

            source (str | None): Source to restrict search to.

        Returns:
            list[tuple[int, Document, float]]: List of tuples containing (id, Document, distance).
        ."""
        # Embed query
        qvec = sqlite_vec.serialize_float32(
            vector=await asyncio.to_thread(self.embed_query, query),
        )

        # Build SQL query
        sql = (
            "SELECT v.chunk_id, c.content, c.metadata, v.distance "
            "FROM vec_chunks v JOIN chunks c ON c.id = v.chunk_id "
            "WHERE v.embedding MATCH :q AND k = :k"
        )

        params = {
            "q": qvec,
            "k": k,
        }

        # Add source filter if provided
        if source is not None:
            sql += " AND v.source = :source"
            params["source"] = source

        # Add distance order
        sql += " ORDER BY v.distance"

        # Execute query
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).fetchall()

        # Return results (triplets of id, document, distance)
        results = []

        for row in rows:
            # Extract row content
            id_, content, metadata, distance = row

            # Create document
            document = Document(
                page_content=content or "",
                metadata=json.loads(
                    metadata or "{}",
                ),
            )

            # Add result
            results.append(
                (
                    id_,
                    document,
                    distance,
                )
            )

        return results

    async def fts_search(
        self,
        query: str,
        k: int,
        source: str | None = None,
    ) -> list[tuple[int, Document, float]]:
        """FTS5 lexical search

        Args:
            query (str): Query string.

            k (int): Number of results to return.

            source (str | None): Source to restrict search to.

        Returns:
            list[tuple[int, Document, float]]: List of tuples containing (id, Document, bm25_rank)."""
        # Build match query
        match_query = " OR ".join(f'"{tok}"' for tok in query.split() if tok)

        # Return empty list if no match query
        if not match_query:
            return []

        # Build SQL query
        sql = (
            "SELECT c.id, c.content, c.metadata, bm25(fts_chunks) AS rank "
            "FROM fts_chunks JOIN chunks c ON c.id = fts_chunks.rowid "
            "WHERE fts_chunks MATCH :q"
        )

        # Build parameters
        params = {"q": match_query, "k": k}

        # Add source filter if provided
        if source is not None:
            sql += " AND c.source = :source"
            params["source"] = source

        sql += " ORDER BY rank LIMIT :k"

        # Execute query
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).fetchall()

        # Return results (triplets of id, document, bm25_rank)
        results = []

        for row in rows:
            # Extract row content
            id_, content, metadata, rank = row

            # Create document
            document = Document(
                page_content=content or "",
                metadata=json.loads(
                    metadata or "{}",
                ),
            )

            # Add result
            results.append(
                (
                    id_,
                    document,
                    rank,
                )
            )

        return results

    async def get_by_function_name(self, function_name: str) -> Document | None:
        """Fetch the full reference chunk for an exact function name."""
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT content, metadata FROM chunks "
                        "WHERE source = 'reference_full' "
                        "AND json_extract(metadata, '$.function_name') = :fn LIMIT 1"
                    ),
                    {"fn": function_name},
                )
            ).fetchone()

        if row is None:
            return None

        return Document(
            page_content=row[0] or "",
            metadata=json.loads(row[1] or "{}"),
        )


def create_vector_store(engine: AsyncEngine, settings: Settings) -> SqliteVecStore:
    """Create the SQLite-vec vector store.

    Args:
        engine (AsyncEngine): SQLAlchemy engine.

        settings (Settings): Application settings.

    Returns:
        SqliteVecStore: The SQLite-vec vector store.
    """
    return SqliteVecStore(engine, settings)
