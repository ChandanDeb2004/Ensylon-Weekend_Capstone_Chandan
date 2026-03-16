# Header-aware Markdown chunking
"""
ingestion/markdown_ingestor.py

Handles ingestion of Markdown documents into semantically coherent chunks.
Documents are split at header boundaries (#, ##, ###) using LangChain's
MarkdownHeaderTextSplitter.  Each chunk is prefixed with its full header
breadcrumb path so that downstream retrieval components retain section
context without needing to inspect a separate metadata object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Headers recognised as split boundaries, ordered from H1 → H3.
# The second element of each tuple is the metadata key LangChain assigns.
# ---------------------------------------------------------------------------
HEADER_SPLITS: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Separator used when joining header levels into a breadcrumb string.
BREADCRUMB_SEPARATOR: str = " > "


def _build_breadcrumb(metadata: dict) -> str:
    """Construct a human-readable breadcrumb string from chunk metadata.

    LangChain's :class:`MarkdownHeaderTextSplitter` populates the
    ``metadata`` dict on each ``Document`` with the currently active
    header at each level.  This function joins those headers in
    hierarchical order to produce a path string like::

        "Legacy Procedures > Procedure X > Manual Steps"

    Args:
        metadata: The ``metadata`` dict from a LangChain ``Document``
            object, keyed by the second element of each tuple in
            :data:`HEADER_SPLITS`.

    Returns:
        A breadcrumb string, or an empty string if no headers are present
        in the metadata (e.g. a Markdown file with no header lines).
    """
    parts: List[str] = []
    for _, key in HEADER_SPLITS:
        value = metadata.get(key, "").strip()
        if value:
            parts.append(value)
    return BREADCRUMB_SEPARATOR.join(parts)


def _prefix_chunk(content: str, breadcrumb: str) -> str:
    """Prepend the section breadcrumb to a chunk's text content.

    Embedding the breadcrumb directly into the chunk string means the
    output contract remains ``List[str]``—no caller needs to handle
    ``Document`` objects or inspect metadata dicts.  The context is
    preserved *inside* the string itself.

    Args:
        content: The raw text body of the chunk.
        breadcrumb: The header path string produced by
            :func:`_build_breadcrumb`.

    Returns:
        If ``breadcrumb`` is non-empty, returns::

            "<breadcrumb>\\n\\n<content>"

        Otherwise returns ``content`` unchanged.
    """
    if not breadcrumb:
        return content
    return f"{breadcrumb}\n\n{content}"


def ingest_markdown(file_path: str | Path) -> List[str]:
    """Ingest a Markdown file and return header-aware text chunks.

    The document is split at ``#``, ``##``, and ``###`` header boundaries
    using :class:`~langchain_text_splitters.MarkdownHeaderTextSplitter`.
    Each resulting chunk is prefixed with its full section breadcrumb
    (e.g. ``"Legacy Procedures > Procedure X"``), so that downstream LLM
    and retrieval components always know which section a chunk originated
    from — critical for conflict detection between documents.

    Chunks whose text content is entirely whitespace after splitting are
    discarded.  If the document contains no headers, the entire file is
    returned as a single chunk.

    Args:
        file_path: Path to the source ``.md`` or ``.markdown`` file.

    Returns:
        A list of strings, each containing an optional breadcrumb prefix
        followed by the chunk's text.  Returns an empty list if the file
        is empty.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist on disk.
        ValueError: If the file extension is not ``.md`` or ``.markdown``.
        RuntimeError: If the file cannot be read (encoding errors, etc.).

    Example:
        >>> chunks = ingest_markdown("docs/operations_wiki.md")
        >>> print(chunks[0])
        Legacy Procedures > Procedure X > Manual Steps
        <BLANKLINE>
        Step 1: Shut down the service before modifying config files.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: '{path}'")

    if path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(
            f"Expected a .md or .markdown file, got '{path.suffix}' "
            f"for path: '{path}'"
        )

    logger.info("Starting Markdown ingestion: %s", path)

    try:
        raw_text: str = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read '{path}': {exc}"
        ) from exc

    if not raw_text.strip():
        logger.warning("'%s' is empty. Returning empty chunk list.", path.name)
        return []

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_SPLITS,
        strip_headers=True,   # Remove the header lines from the body text;
                              # they are preserved in metadata / breadcrumb.
    )

    documents = splitter.split_text(raw_text)
    logger.info(
        "MarkdownHeaderTextSplitter produced %d document(s) from '%s'.",
        len(documents),
        path.name,
    )

    chunks: List[str] = []
    for doc in documents:
        body: str = doc.page_content.strip()
        if not body:
            # Skip header-only sections with no body text.
            continue

        breadcrumb: str = _build_breadcrumb(doc.metadata)
        chunk: str = _prefix_chunk(body, breadcrumb)
        chunks.append(chunk)

    logger.info(
        "Ingestion complete: %d non-empty chunk(s) extracted from '%s'.",
        len(chunks),
        path.name,
    )
    return chunks