#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Agent tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pydash as py_
from langchain_core.tools import tool

from sitsrag.logging import get_logger
from sitsrag.ontology.expander import (
    format_collection_chunk,
    format_index_chunk,
    format_satellite_chunk,
    search_collections,
    search_indices,
    search_satellites,
)

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from sitsrag.config import Settings
    from sitsrag.db.graph.vector_store import SqliteVecStore
    from sitsrag.observability import TraceSpan
    from sitsrag.providers.reranker import Reranker


#
# Logger
#
logger = get_logger(__name__)


#
# Auxiliary functions
#
def _format_results(
    results: list[tuple],
    max_chunks: int = 5,
) -> tuple[str, dict]:
    """Format reranked tuples as agent content plus source metadata.

    Args:
        results (list[tuple]): List of reranked tuples.

        max_chunks (int): Maximum number of chunks to return.

    Returns:
        tuple[str, dict]: Tuple containing the formatted content and source metadata.
    """
    if not results:
        return "No results found.", {"sources": {}}

    # Initialize parts
    parts = []
    sources = {}

    # Track markers already to avoid duplicates
    seen = {}
    marker_index = 0

    for doc, _score in results[:max_chunks]:
        # Get metadata
        meta = doc.metadata

        # Get source
        source = py_.get(meta, "source", "unknown")

        # Get title
        title = (
            py_.chain(["section_title", "function_name", "filename", "title"])
            .map_(lambda k: py_.get(meta, k))
            .compact()
            .head()
            .value()
        ) or "Unknown"

        # Get URL
        doi = py_.get(meta, "doi")
        url = (
            py_.get(meta, "page_url")
            or (f"https://doi.org/{doi}" if doi else None)
            or py_.get(meta, "url")
        ) or None

        # Deduplicate by the source identity (URL, else title)
        key = url or title

        # Reuse the existing marker for a duplicate, otherwise mint a new one
        if key in seen:
            marker = seen[key]

        else:
            marker_index += 1
            marker = f"S{marker_index}"

            # Add to seen
            seen[key] = marker

            sources[marker] = {
                "title": title,
                "url": url,
                "source_type": source,
            }

        # Add header and content to parts (duplicates add context under same marker)
        parts.append(f"[{marker}] {title}\n{doc.page_content}")

    # Join parts with separator
    return "\n\n---\n\n".join(parts), {"sources": sources}


#
# High-level interface
#
def make_search_documentation_tool(
    retriever: BaseRetriever,
    reranker: Reranker,
    settings: Settings,
    trace_span: TraceSpan,
):
    """Create a tool that searches the SITS book documentation."""

    @tool(response_format="content_and_artifact")
    async def search_documentation(query: str) -> tuple[str, dict]:
        """Search the SITS book documentation.

        Use this when user wants to learn concepts, workflows and details
        about SITS methods and tools. Examples of topics covered here are:
        - Classification workflow
        - Data Cube creation workflow
        - Conceptual explanations and details about SITS and its tools
        - Concepts
        - Usage patterns
        - Machine Learning Models available in SITS
        - Deep Learning models available in SITS
        - Details about methods available in SITS (e.g., Self-Organized Maps - SOM, balance, Sampling)

        Args:
            query: Natural language query.
        """
        # Search documentation
        async with trace_span("tool:search_documentation", query=query[:80]):
            # Invoke retriever
            docs = await retriever.ainvoke(query)

            # Build results
            results = [(doc, 0.0) for doc in docs]

            # Rerank results
            reranked = await reranker.rerank(
                query,
                results,
                top_k=settings.rerank_top_k,
            )

        # Format results
        return _format_results(reranked)

    return search_documentation


def make_search_reference_tool(
    retriever: BaseRetriever,
    reranker: Reranker,
    settings: Settings,
    trace_span: TraceSpan,
):
    """Create a tool that searches the SITS function reference."""

    @tool(response_format="content_and_artifact")
    async def search_reference(query: str) -> tuple[str, dict]:
        """Search the SITS functions reference.

        Use this when the user asks about a specific function or needs exacts
        parameter names. All functions available in SITS are available here, including:
        - sits_cube
        - sits_som_map
        - sits_select
        - sits_regularize
        - sits_label_classification

        Args:
            query: Natural language query.
        """

        # Search reference
        async with trace_span("tool:search_reference", query=query[:80]):
            # Invoke retriever
            docs = await retriever.ainvoke(query)

            # Build results
            results = [(doc, 0.0) for doc in docs]

            # Rerank results
            reranked = await reranker.rerank(
                query,
                results,
                top_k=settings.rerank_top_k,
            )

        # Format results
        return _format_results(reranked)

    # Return tool
    return search_reference


def make_get_function_detail_tool(
    vector_store: SqliteVecStore,
):
    """Create a tool that retrieves the complete documentation for a function."""

    @tool(response_format="content_and_artifact")
    async def get_function_detail(
        function_name: str,
    ) -> tuple[str, dict[str, dict[str, dict[str, str | None]]]]:
        """Retrieve the COMPLETE documentation for a SITS function.

        Use this when you need complete documentation of a SITS function. Here,
        we include everything about a function, including:
        - Full signature
        - All parameters supported
        - Return value / format
        - Usage examples

        Use this after `search_reference` returns partial results and you need
        the full picture to answer accurately

        Args:
            function_name: Name of the function to be described.
        """

        # Exact lookup of the full reference chunk by function name
        doc = await vector_store.get_by_function_name(function_name)

        # Check if a match was found
        if doc is None:
            return (
                f"No complete documentation found for function '{function_name}'.",
                {"sources": {}},
            )

        # Get URL
        url = doc.metadata.get("page_url", "")

        content = f"[S1] {function_name}\n\n{doc.page_content}"
        artifact = {
            "sources": {
                "S1": {
                    "title": function_name,
                    "url": url or None,
                    "source_type": "reference_full",
                }
            }
        }

        return content, artifact

    # Return tool
    return get_function_detail


def make_search_articles_tool(
    retriever: BaseRetriever,
    reranker: Reranker,
    settings: Settings,
    trace_span: TraceSpan,
):
    """Create a tool that searches published research articles."""

    @tool(response_format="content_and_artifact")
    async def search_articles(query: str) -> tuple[str, dict]:
        """Search published research articles that use the SITS package.

        Use this when user wants to know:
        - What are the articles available using SITS
        - How people are using SITS to produce scientific and governamental data
        - Examples of real-world applications using SITS
        - Case studies using SITS
        - Published results using SITS

        Args:
            query: Natural language query.
        """

        # Search articles
        async with trace_span("tool:search_articles", query=query[:80]):
            # Invoke retriever
            docs = await retriever.ainvoke(query)

            # Build results
            results = [(doc, 0.0) for doc in docs]

            # Rerank results
            reranked = await reranker.rerank(
                query,
                results,
                top_k=settings.rerank_top_k,
            )

        # Format results
        return _format_results(reranked)

    # Return tool
    return search_articles


def make_lookup_spectral_index_tool(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Create a tool that looks up spectral indices."""

    @tool(response_format="content_and_artifact")
    async def lookup_spectral_index(name_or_keyword: str) -> tuple[str, dict]:
        """Get spectral indices information, by name or keyword.

        Use this when user is asking about or wants to use a spectral indice. This
        includes cases like:
        - Wants to know about a spectral indice (vegetation, water, burn);
        - Wants to apply a spectral indice using SITS (e.g., NDVI, EVI, NDWI, NDWI2, NBR)

        This is a complete database of indices, so this returns:
        - Formula
        - Required band types
        - Application domain

        Args:
            name_or_keyword: Name or keyword of an index (full-text search supported)
        """
        # Search indices
        async with session_factory() as session:
            indices = await search_indices(session, name_or_keyword)

        # Check if results are empty
        if not indices:
            return f"No spectral index found matching '{name_or_keyword}'.", {"sources": {}}

        # Format results
        return "\n\n".join(format_index_chunk(idx) for idx in indices), {"sources": {}}

    # Return tool
    return lookup_spectral_index


def make_lookup_satellite_tool(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Create a tool that looks up satellite information."""

    @tool(response_format="content_and_artifact")
    async def lookup_satellite(name_or_keyword: str) -> tuple[str, dict]:
        """Get a satellite and its details, including spectral bands and spatial resolution.

        Use this when the user mentions a satellite like Sentinel-2, Landsat-8,
        MODIS. This can provide complementary information for the understanding
        of SITS functions, usage of spectral indices and more.

        This returns:
        - Satellite name
        - Sensors metadata (e.g., waveband_region, bound_min, bound_max, resolution, bands)

        If user talks about computable indices using a satellite, you can create connections
        of `computable_indices` using `lookup_spectral_index`

        Args:
            name_or_keyword: Name or keyword of an index (full-text search supported)
        """

        # Search satellites
        async with session_factory() as session:
            sats = await search_satellites(session, name_or_keyword)

        # Check if results are empty
        if not sats:
            return f"No satellite found matching '{name_or_keyword}'.", {"sources": {}}

        # Format satellites
        return "\n\n".join(format_satellite_chunk(sat) for sat in sats), {"sources": {}}

    # Return tool
    return lookup_satellite


def make_lookup_collection_tool(
    session_factory: async_sessionmaker[AsyncSession],
):
    """Create a tool that looks up SITS data collections."""

    @tool(response_format="content_and_artifact")
    async def lookup_collection(name_or_keyword: str) -> tuple[str, dict]:
        """Look up a SITS data collection to find its EXACT band names, time period, and data source.

        ALWAYS use this before writing code that references band names. The band
        names returned here are the ONLY valid identifiers for code.

        Args:
            name_or_keyword: Name or keyword of an index (full-text search supported)
        """

        # Search collections
        async with session_factory() as session:
            colls = await search_collections(session, name_or_keyword)

        # Check if results are empty
        if not colls:
            return f"No collection found matching '{name_or_keyword}'.", {"sources": {}}

        # Format collections
        return "\n\n".join(format_collection_chunk(c) for c in colls), {"sources": {}}

    # Return tool
    return lookup_collection
