#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Reranker."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from sitsrag.logging import get_logger

if TYPE_CHECKING:
    from sitsrag.config import Settings


#
# Logger
#
logger = get_logger(__name__)


#
# Protocols
#
class Reranker(Protocol):
    """Abstract interface for reranking backends."""

    async def rerank(
        self,
        query: str,
        results: list[tuple[Document, float]],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Rerank results by relevance to the query.

        Args:
            query (str): The query to rerank the results by relevance.

            results (list[tuple[Document, float]]): The list of results to rerank.

            top_k (int): The number of top results to return.

        Returns:
            list[tuple[Document, float]]: The reranked results.
        """
        ...


#
# Providers
#
class CohereReranker(Reranker):
    """Rerank search results using Cohere's hosted Rerank API.

    Reranking is a network call, so it runs off the box's CPU entirely. Unlike a
    local cross-encoder (which saturates every core and caps throughput), it does
    not compete for compute, letting a single worker sustain many concurrent
    requests — the rerank is just an I/O wait the event loop absorbs cheaply.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the CohereReranker."""
        self._client = CohereRerank(
            cohere_api_key=settings.reranker_api_key,
            model=settings.reranker_model,
            top_n=settings.rerank_top_k,
        )

    async def rerank(
        self,
        query: str,
        results: list[tuple[Document, float]],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        if not results:
            return []

        # Build passages — Cohere reranks raw text and returns indices into this list
        documents = [doc.page_content for doc, _score in results]

        # Get event loop
        loop = asyncio.get_event_loop()

        # Rerank! (network call, offloaded so the event loop stays free). The
        # client returns [{"index", "relevance_score"}] already sorted by relevance.
        try:
            ranked = await loop.run_in_executor(
                None,
                lambda: self._client.rerank(documents=documents, query=query, top_n=top_k),
            )

        # Never let a reranker outage (429, timeout, network) break retrieval —
        # fall back to the retriever's original RRF-fused order.
        except Exception as exc:  # noqa: BLE001 - degrade on any provider failure
            logger.warning(
                event="rerank.fallback",
                error=str(exc),
                kept=min(top_k, len(results)),
            )
            return results[:top_k]

        # Map Cohere's scores back onto the original documents by index
        reranked = [(results[item["index"]][0], item["relevance_score"]) for item in ranked]

        # Log
        logger.info(
            event="Reranked results",
            before=len(results),
            after=len(reranked),
        )

        # Return!
        return reranked


def build_reranker(settings: Settings) -> Reranker:
    """Build a reranker based on the configured provider.

    Args:
        settings (Settings): Application settings.

    Returns:
        Reranker: A reranker instance implementing the ``Reranker`` protocol.
    """
    # ToDo: Can be extended to support multiple rerankers
    return CohereReranker(settings)
