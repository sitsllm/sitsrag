#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Parser for the SITS function reference."""

import hashlib
from xml.etree import ElementTree

import httpx
import pydash as py_
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document

from sitsrag.logging import get_logger
from sitsrag.parsers.chunker import chunk_text

#
# Logger
#
logger = get_logger(__name__)


#
# Auxiliary functions
#
async def _fetch_reference_urls(base_url: str) -> list[str]:
    """Get function page URLs from sitemap, filtering to reference pages.

    Args:
        base_url (str): Base URL of the SITS reference.

    Returns:
        list[str]: List of URLs.
    """
    # Build sitemap URL
    sitemap_url = f"{base_url}/sitemap.xml"

    # Fetch sitemap
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(sitemap_url)
        response.raise_for_status()

    # Parse sitemap
    root = ElementTree.fromstring(response.text)

    # Build namespace
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # Find all loc elements
    urls = py_.filter_(
        py_.compact(
            py_.map_(
                root.findall(
                    ".//sm:loc",
                    ns,
                ),
                lambda loc: loc.text,
            ),
        ),
        lambda u: "/reference/" in u and not u.endswith("/reference/index.html"),
    )

    logger.info(
        event="Found reference pages in sitemap",
        count=len(urls),
    )

    # Return
    return urls


def _build_chunk_prefix(
    function_name: str | None,
    url: str,
    section: str | None = None,
) -> str:
    """Build a context prefix for a reference chunk.

    Args:
        function_name (str | None): The function name.

        url (str): The documentation URL.

        section (str | None): Section name.

    Returns:
        str: Prefix string.
    """
    return "\n".join(
        py_.compact(
            [
                function_name and f"Function: {function_name}",
                f"Docs: {url}",
                section and f"{section}:",
            ]
        )
    )


def _parse_reference_page(
    url: str, html: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Extract structured sections from a pkgdown function page.

    Args:
        url (str): URL of the page.

        html (str): HTML content of the page.

        chunk_size (int): Maximum size of each chunk.

        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list[Document]: List of document chunks.
    """
    # Parse HTML
    soup = BeautifulSoup(html, "lxml")

    # Find main content
    main = soup.find("main") or soup.find("div", {"role": "main"})

    # Check if no main content is found
    if not main:
        return []

    # Extract function name
    function_name = _extract_function_name(url)

    # Extract reference sections
    sections = _extract_reference_sections(main)

    # Build combined text for the function (with context prefix)
    prefix = _build_chunk_prefix(function_name, url)
    full_text_parts = [prefix]

    # Iterate over the sections
    for section_name, section_text in sections:
        # If section text is empty, skip it
        if section_text.strip():
            full_text_parts.append(f"{section_name}:\n{section_text}")

    # Join full text parts
    full_text = "\n\n".join(full_text_parts)

    # If there is only one part, return empty list
    if len(full_text_parts) <= 1:
        return []

    # Define full chunk ID
    full_chunk_id = hashlib.sha256(f"{url}:reference_full".encode()).hexdigest()[:16]

    # Add full chunk
    chunks = []
    chunks.append(
        Document(
            page_content=full_text,
            metadata={
                "source": "reference_full",
                "function_name": function_name,
                "page_url": url,
                "section": "full",
                "chunk_id": f"ref-full-{full_chunk_id}",
            },
        )
    )

    # Check if function fits in one chunk
    if len(full_text) <= chunk_size:
        # Define chunk ID
        chunk_id = hashlib.sha256(f"{url}:full".encode()).hexdigest()[:16]

        # Add chunk
        chunks.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": "reference",
                    "function_name": function_name,
                    "page_url": url,
                    "section": "full",
                    "chunk_id": f"ref-{chunk_id}",
                },
            )
        )

    else:
        # Split into section-based chunks
        for section_name, section_text in sections:
            # If section text is empty, skip it
            if not section_text.strip():
                continue

            # Build section text
            header = _build_chunk_prefix(
                function_name=function_name,
                url=url,
                section=section_name,
            )

            text = f"{header}\n{section_text}"

            # Split text into chunks
            text_chunks = chunk_text(text, chunk_size, chunk_overlap)

            # Iterate over chunks
            for i, t in enumerate(text_chunks):
                # Generate a chunk ID
                chunk_id = hashlib.sha256(f"{url}:{section_name}:{i}".encode()).hexdigest()[:16]

                # Add the chunk to the list of chunks
                chunks.append(
                    Document(
                        page_content=t,
                        metadata={
                            "source": "reference",
                            "function_name": function_name,
                            "page_url": url,
                            "section": section_name.lower(),
                            "chunk_id": f"ref-{chunk_id}",
                        },
                    )
                )

    # Return
    return chunks


def _extract_function_name(url: str) -> str | None:
    """Derive the function name from the pkgdown page slug.

    Args:
        url (str): URL of the reference page.

    Returns:
        str | None: The function name.
    """
    # Take the page slug
    slug = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")

    # pkgdown encodes a leading dot (internal functions) as a "dot-" prefix
    if slug.startswith("dot-"):
        slug = f".{slug[4:]}"

    # Return
    return slug or None


def _extract_reference_sections(main: Tag) -> list[tuple[str, str]]:
    """Extract named sections (Usage, Arguments, Value, Examples, etc.).

    Args:
        main (Tag): The main content tag.

    Returns:
        list[tuple[str, str]]: The list of sections.
    """
    sections = []

    # Find all `h2` and `h3` tags
    # pkgdown uses `<h2>` for section headers inside main
    headings = main.find_all(["h2", "h3"])

    # Iterate over headings
    for heading in headings:
        section_name = heading.get_text(strip=True)

        # Initialize list of content parts
        content_parts = []

        # Iterate over siblings
        for sibling in heading.find_next_siblings():
            # If sibling is not a tag, skip it
            if not isinstance(sibling, Tag):
                continue

            # If sibling is an h2 or h3 heading, break
            if sibling.name in ("h2", "h3"):
                break

            # Get text of sibling
            text = sibling.get_text(separator=" ", strip=True)

            # If text is not empty, append it to content parts
            if text:
                content_parts.append(text)

        # If content parts are not empty, add section
        if content_parts:
            sections.append((section_name, "\n".join(content_parts)))

    # Initialize list of description parts
    desc_parts = []

    # Iterate over children
    for child in main.children:
        # If child is not a tag, skip it
        if not isinstance(child, Tag):
            continue

        # If child is an `h1`, `h2`, or `h3` heading, break
        if child.name in ("h1", "h2", "h3"):
            break

        # Get the text of the child
        text = child.get_text(separator=" ", strip=True)

        # If the text is not empty, add it to the description parts
        if text:
            desc_parts.append(text)

    # If description parts are not empty, add description to sections
    if desc_parts:
        sections.insert(0, ("Description", "\n".join(desc_parts)))

    # Return
    return sections


async def parse_reference(
    base_url: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Fetch all SITS reference pages and parse them into document chunks.

    Args:
        base_url (str): The base URL of the SITS reference.

        chunk_size (int): The maximum size of each chunk.

        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        list[Document]: The list of document chunks.
    """
    # Fetch the URLs from the sitemap
    urls = await _fetch_reference_urls(base_url)

    # Iterate over the URLs
    all_chunks = []

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for url in urls:
            try:
                # Fetch the URL
                response = await client.get(url)

                # Raise for status
                response.raise_for_status()

                # Parse the page
                chunks = _parse_reference_page(
                    url=url,
                    html=response.text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Add chunks
                all_chunks.extend(chunks)

                # Log
                logger.debug(
                    event="Parsed chunks",
                    count=len(chunks),
                    url=url,
                )

            # Handle HTTP errors
            except httpx.HTTPError as e:
                logger.warning(
                    event="Failed to fetch reference page",
                    url=url,
                    error=str(e),
                )

    # Log
    logger.info(
        event="Total reference chunks",
        count=len(all_chunks),
    )

    # Return
    return all_chunks
