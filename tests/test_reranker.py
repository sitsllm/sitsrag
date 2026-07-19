#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Tests for the Cohere reranker provider (mocked)."""

from unittest.mock import patch

from langchain_core.documents import Document

from sitsrag.config import Settings
from sitsrag.providers.reranker import CohereReranker, build_reranker


def _settings() -> Settings:
    """Settings with a (fake) reranker key so the client can be constructed."""
    return Settings(
        reranker_api_key="fake-key",
        reranker_model="rerank-v3.5",
        rerank_top_k=5,
    )


def _results(n: int) -> list[tuple[Document, float]]:
    """Build ``n`` (Document, score) pairs in retriever (RRF) order."""
    return [(Document(page_content=f"doc{i}"), 0.0) for i in range(n)]


def test_build_reranker_returns_cohere():
    """The factory wires the Cohere reranker."""
    with patch("sitsrag.providers.reranker.CohereRerank"):
        reranker = build_reranker(_settings())

    assert isinstance(reranker, CohereReranker)


async def test_rerank_empty_returns_empty():
    """Reranking an empty result set is a no-op (and makes no client call)."""
    with patch("sitsrag.providers.reranker.CohereRerank") as mock_cls:
        reranker = build_reranker(_settings())

        out = await reranker.rerank("query", [], top_k=5)

    # Assert
    assert out == []
    mock_cls.return_value.rerank.assert_not_called()


async def test_rerank_maps_scores_by_index():
    """Cohere (index, relevance_score) pairs map back onto the original documents."""
    with patch("sitsrag.providers.reranker.CohereRerank") as mock_cls:
        client = mock_cls.return_value

        # Cohere returns indices into the input list (already sorted by relevance)
        client.rerank.return_value = [
            {"index": 2, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.42},
        ]

        reranker = build_reranker(_settings())
        results = _results(3)

        out = await reranker.rerank("query", results, top_k=2)

    # The reranked docs and scores follow Cohere order/indices
    assert [doc.page_content for doc, _ in out] == ["doc2", "doc0"]
    assert [score for _, score in out] == [0.91, 0.42]

    # The original documents are preserved (identity), not copies
    assert out[0][0] is results[2][0]
    assert out[1][0] is results[0][0]

    # top_k is forwarded to Cohere as top_n, and all candidates are sent
    _, kwargs = client.rerank.call_args

    assert kwargs["top_n"] == 2
    assert kwargs["query"] == "query"
    assert kwargs["documents"] == ["doc0", "doc1", "doc2"]


async def test_rerank_falls_back_to_rrf_order_on_error():
    """A provider failure degrades to the retriever original order (no exception)."""
    with patch("sitsrag.providers.reranker.CohereRerank") as mock_cls:
        mock_cls.return_value.rerank.side_effect = RuntimeError("429 Too Many Requests")

        reranker = build_reranker(_settings())
        results = _results(4)

        out = await reranker.rerank("query", results, top_k=2)

    # Falls back to the first `top_k` in the incoming (RRF fused) order
    assert [doc.page_content for doc, _ in out] == ["doc0", "doc1"]
    assert out[0][0] is results[0][0]
    assert out[1][0] is results[1][0]


async def test_rerank_fallback_respects_available_count():
    """Fallback never returns more than the available results."""
    with patch("sitsrag.providers.reranker.CohereRerank") as mock_cls:
        mock_cls.return_value.rerank.side_effect = ConnectionError("network down")

        reranker = build_reranker(_settings())
        results = _results(2)

        out = await reranker.rerank("query", results, top_k=5)

    # Only two candidates exist, so only two come back
    assert len(out) == 2
    assert [doc.page_content for doc, _ in out] == ["doc0", "doc1"]
