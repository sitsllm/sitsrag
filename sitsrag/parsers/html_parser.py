#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""HTML parser."""

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
# Utilities
#
def _parse_page(url: str, html: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Extract and chunk the main content from a Quarto HTML page.

    Args:
        url (str): The URL of the page.

        html (str): The HTML content of the page.

        chunk_size (int): The maximum size of each chunk.

        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        list[Document]: The list of document chunks.
    """
    # Parse the HTML
    soup = BeautifulSoup(html, "lxml")

    # Find the main content
    main = soup.find("main") or soup.find("div", {"class": "content"})

    # If no main content is found, return an empty list
    if not main:
        return []

    # Extract the chapter title
    chapter = _extract_chapter_title(soup)

    # Extract the sections
    sections = _split_into_sections(main)

    # Build context prefix for every chunk
    context_prefix = (
        "\n".join(
            py_.compact(
                [
                    chapter and f"Article: {chapter}",
                    f"URL: {url}",
                ]
            )
        )
        + "\n\n"
    )

    # Initialize the list of chunks
    chunks = []

    for section_title, section_anchor, section_text in sections:
        # If the section text is empty, skip it
        if not section_text.strip():
            continue

        # Deep-link to the section anchor when Quarto provides one (falls back to the page)
        page_url = f"{url}#{section_anchor}" if section_anchor else url

        # Split the section text into chunks
        text_chunks: list[str] = chunk_text(
            text=section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Iterate over text chunks
        for i, text in enumerate(text_chunks):
            # Generate chunk ID
            chunk_id = hashlib.sha256(f"{url}:{section_title}:{i}".encode()).hexdigest()[:16]

            # Prepend context prefix so every chunk carries its source identity
            prefixed_text = f"{context_prefix}{text}"

            # Add chunk
            chunks.append(
                Document(
                    page_content=prefixed_text,
                    metadata={
                        "source": "documentation",
                        "page_url": page_url,
                        "section_title": section_title or chapter,
                        "chapter": chapter,
                        "chunk_id": f"doc-{chunk_id}",
                    },
                )
            )

    # Return
    return chunks


def _extract_chapter_title(soup: BeautifulSoup) -> str | None:
    """Extract the page/chapter title.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object.

    Returns:
        str | None: The chapter title.
    """
    # Get candidates (h1 preferred, page <title> as fallback)
    candidates = py_.compact([soup.find("h1"), soup.find("title")])

    # Get first candidate
    first = py_.head(candidates)

    # Return (separator keeps Quarto's glued section number apart from the text)
    return first.get_text(separator=" ", strip=True) if first else None


def _split_into_sections(main: Tag) -> list[tuple[str | None, str | None, str]]:
    """Split main content into sections based on ``h2`` and ``h3`` headings.

    Quarto wraps every heading inside a ``<section class="levelN">`` element, so the
    headings are never direct children of ``<main>``. We walk each ``<section>`` and read
    its own `heading + own text`. Nested subsections are visited on their own iteration, so
    their text is not double-counted here. Each section also carries the ``id`` Quarto emits,
    so callers can deep-link to the exact section anchor.

    Args:
        main (Tag): The main content tag.

    Returns:
        list[tuple[str | None, str | None, str]]: The list of ``(title, anchor, text)`` sections.
    """
    # Initialize list of sections
    sections = []

    # Iterate over the section wrappers Quarto emits
    for section in main.find_all("section"):
        # Section title is this section heading (first h1-h4 direct child)
        heading = section.find(["h1", "h2", "h3", "h4"], recursive=False)

        # Get title
        current_title = heading.get_text(separator=" ", strip=True) if heading else None

        # Section anchor (Quarto sets the id on the <section> for deep-linking)
        current_anchor = section.get("id")

        # Collect text from direct children, skipping the heading and nested sections
        current_texts = []

        for child in section.find_all(recursive=False):
            if child is heading or child.name == "section":
                continue

            # Get text of element
            text = child.get_text(separator=" ", strip=True)

            # Check if text is not empty
            if text:
                current_texts.append(text)

        # Append the section if it holds any text
        if current_texts:
            sections.append((current_title, current_anchor, "\n\n".join(current_texts)))

    # Return
    return sections


#
# High-level interface
#
async def fetch_sitemap_urls(base_url: str) -> list[str]:
    """Fetch and parse ``sitemap.xml``.

    Args:
        base_url (str): Base URL of the SITS book.

    Returns:
        list[str]: The list of URLs.
    """
    # Construct the sitemap URL
    sitemap_url = f"{base_url}/sitemap.xml"

    # Fetch the sitemap
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(sitemap_url)
        response.raise_for_status()

    # Parse the sitemap
    root = ElementTree.fromstring(response.text)

    # Build namespace
    ns = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    }

    # Find all loc elements
    urls = py_.compact(
        py_.map_(
            root.findall(
                ".//sm:loc",
                ns,
            ),
            lambda loc: loc.text,
        ),
    )

    # Log
    logger.info(
        event="Found URLs in sitemap",
        count=len(urls),
        sitemap_url=sitemap_url,
    )

    # Return!
    return urls


async def parse_documentation(
    base_url: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Fetch all SITS book pages and parse them into document chunks.

    Args:
        base_url (str): Base URL of the SITS book.

        chunk_size (int): Maximum size of each chunk.

        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list[Document]: List of document chunks.
    """
    # Fetch URLs from sitemap
    urls = await fetch_sitemap_urls(base_url)

    # Initialize list of document chunks
    all_chunks = []

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for url in urls:
            try:
                # Fetch the URL
                response = await client.get(url)

                # Raise for status
                response.raise_for_status()

                # Parse the page
                chunks = _parse_page(
                    url=url,
                    html=response.text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Add the chunks to the list of document chunks
                all_chunks.extend(chunks)

                # Log the number of chunks parsed
                logger.info(
                    event="Parsed chunks",
                    count=len(chunks),
                    url=url,
                )

            # Handle HTTP errors
            except httpx.HTTPError as e:
                logger.warning("Failed to fetch page", url=url, error=str(e))

    logger.info(
        event="Total documentation chunks",
        count=len(all_chunks),
    )

    # Return
    return all_chunks
