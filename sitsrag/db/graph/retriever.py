#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Hybrid retriever module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from sitsrag.logging import get_logger

if TYPE_CHECKING:
    from sitsrag.config import Settings
    from sitsrag.db.graph.vector_store import SqliteVecStore

#
# Logger
#
logger = get_logger(__name__)

#
# Reciprocal Rank Fusion constant.
#
RRF_K = 60


#
# Hybrid retriever class.
#
class HybridRetriever(BaseRetriever):
    """Fuse sqlite-vec KNN and FTS5 lexical results with RRF."""

    k: int = 20
    """Number of results to return."""

    source: str | None = None
    """Restrict search to a single content source (documentation/reference/pdf)."""

    vector_weight: float = 0.7
    """Weight for the vector search."""

    fts_weight: float = 0.3
    """Weight for the FTS5 search."""

    _store: Any = PrivateAttr()
    """Vector store."""

    def __init__(self, store: SqliteVecStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._store = store

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        raise NotImplementedError("Use ainvoke() - this project is fully async.")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        vector_hits = await self._store.vector_search(query=query, k=self.k, source=self.source)
        fts_hits = await self._store.fts_search(query=query, k=self.k, source=self.source)

        # Implement RRF by chunk id.
        docs = {}
        scores = {}

        # Iterate vector hits
        for rank, (cid, doc, _dist) in enumerate(vector_hits):
            # Update scores
            scores[cid] = scores.get(cid, 0.0) + self.vector_weight / (RRF_K + rank)

            # Update docs
            docs[cid] = doc

        # Iterate FTS hits
        for rank, (cid, doc, _rank) in enumerate(fts_hits):
            # Update scores
            scores[cid] = scores.get(cid, 0.0) + self.fts_weight / (RRF_K + rank)

            # Update docs
            docs.setdefault(cid, doc)

        # Order by score
        ordered = sorted(
            scores,
            key=lambda cid: scores[cid],
            reverse=True,
        )
        ordered = ordered[: self.k]

        # Log results
        logger.debug(
            "hybrid retriever results",
            query=query[:80],
            vector=len(vector_hits),
            fts=len(fts_hits),
            fused=len(ordered),
        )

        # Return docs
        return [docs[cid] for cid in ordered]


def create_hybrid_retriever(
    store: SqliteVecStore,
    settings: Settings,
    *,
    source: str | None = None,
) -> HybridRetriever:
    """Create a hybrid retriever for a single content source."""
    return HybridRetriever(
        store,
        k=settings.retrieval_top_k,
        source=source,
        vector_weight=settings.vector_weight,
        fts_weight=settings.fts_weight,
    )
