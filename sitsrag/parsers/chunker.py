#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""Text splitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks.

    Splits on paragraph breaks first, then newlines, then sentences, then spaces.
    Each chunk is at most `chunk_size` characters, with `chunk_overlap` overlap.

    Args:
        text (str): The text to split.

        chunk_size (int): The maximum size of each chunk.

        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        list[str]: The list of chunks.
    """
    if not text or not text.strip():
        return []

    # Define splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split the text into chunks
    return splitter.split_text(text.strip())
