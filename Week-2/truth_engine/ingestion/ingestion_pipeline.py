"""
ingestion/ingestion_pipeline.py

Orchestrates the full document ingestion pipeline.

Calls each domain-specific ingestor (PDF, JSON/CSV, Markdown), merges
their outputs into a single unified chunk list, and performs MD5-based
deduplication to remove exact-copy content that may appear across multiple
source documents.

This is the single entry point that ``main.py`` calls to trigger the
entire ingestion process.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, NamedTuple

from ingestion.json_ingestor import ingest_json
from ingestion.markdown_ingestor import ingest_markdown
from ingestion.pdf_ingestor import ingest_pdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class IngestionResult(NamedTuple):
    """Structured result returned by :func:`run_ingestion_pipeline`.

    Attributes:
        chunks: The deduplicated, merged list of text chunks ready for
            embedding and storage.
        total_raw: Total number of chunks collected across all ingestors
            before deduplication.
        duplicates_removed: Number of exact-duplicate chunks discarded
            during the deduplication pass.
        failed_sources: List of file path strings for which ingestion
            raised an exception.  A non-empty list indicates partial
            ingestion — the pipeline continued with the remaining sources.
    """

    chunks: List[str]
    total_raw: int
    duplicates_removed: int
    failed_sources: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _md5(text: str) -> str:
    """Compute the MD5 hex digest of a normalised text string.

    Whitespace is stripped before hashing so that semantically identical
    chunks that differ only in leading/trailing whitespace are treated as
    duplicates.

    Args:
        text: The chunk string to hash.

    Returns:
        A 32-character lowercase hexadecimal MD5 digest string.
    """
    normalised = text.strip()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def _deduplicate(chunks: List[str]) -> tuple[List[str], int]:
    """Remove exact-duplicate chunks using MD5 content hashing.

    Preserves the first occurrence of each unique chunk and discards
    subsequent duplicates.  Order of the first-seen chunks is maintained.

    Args:
        chunks: The raw merged list of chunks from all ingestors.

    Returns:
        A 2-tuple of:
        - The deduplicated list of chunks.
        - The integer count of chunks that were removed.
    """
    seen: set[str] = set()
    unique: List[str] = []

    for chunk in chunks:
        digest = _md5(chunk)
        if digest not in seen:
            seen.add(digest)
            unique.append(chunk)

    duplicates_removed = len(chunks) - len(unique)
    return unique, duplicates_removed


def _safe_ingest(label: str, ingest_fn, *args, **kwargs) -> List[str]:
    """Call an ingestor function, returning an empty list on failure.

    Isolates ingestor exceptions so that a failure in one source does
    not abort ingestion of the remaining sources.  The exception is
    logged at ERROR level for observability.

    Args:
        label: A human-readable source identifier for log messages
            (e.g. the file path string).
        ingest_fn: The ingestor callable to invoke.
        *args: Positional arguments forwarded to ``ingest_fn``.
        **kwargs: Keyword arguments forwarded to ``ingest_fn``.

    Returns:
        The list of chunks returned by ``ingest_fn``, or an empty list
        if the call raised any exception.
    """
    try:
        return ingest_fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Ingestion failed for source '%s': %s — skipping.",
            label,
            exc,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ingestion_pipeline(
    pdf_paths: List[str | Path] | None = None,
    json_paths: List[str | Path] | None = None,
    markdown_paths: List[str | Path] | None = None,
    json_template: str | None = None,
) -> IngestionResult:
    """Run the full document ingestion pipeline and return deduplicated chunks.

    Orchestrates :func:`~ingestion.pdf_ingestor.ingest_pdf`,
    :func:`~ingestion.json_ingestor.ingest_json`, and
    :func:`~ingestion.markdown_ingestor.ingest_markdown` across all
    provided source files.  Results from all ingestors are merged into a
    single list and deduplicated via MD5 content hashing before being
    returned.

    If an individual source file fails to ingest (file not found, parse
    error, etc.), the error is logged and that source is skipped.  The
    pipeline continues with the remaining sources rather than aborting.

    Args:
        pdf_paths: Optional list of paths to ``.pdf`` files.
        json_paths: Optional list of paths to ``.json`` or ``.csv`` files.
        markdown_paths: Optional list of paths to ``.md`` or ``.markdown``
            files.
        json_template: Optional template string passed to
            :func:`~ingestion.json_ingestor.ingest_json` for all JSON/CSV
            sources.  Falls back to
            :data:`~ingestion.json_ingestor.DEFAULT_TEMPLATE` if omitted.

    Returns:
        An :class:`IngestionResult` named tuple containing:

        - ``chunks``: The final, deduplicated ``List[str]`` ready for
          embedding.
        - ``total_raw``: Total chunks before deduplication.
        - ``duplicates_removed``: Number of duplicates discarded.
        - ``failed_sources``: Paths of any sources that raised exceptions.

    """
    pdf_paths = pdf_paths or []
    json_paths = json_paths or []
    markdown_paths = markdown_paths or []

    all_chunks: List[str] = []
    failed_sources: List[str] = []

    # ------------------------------------------------------------------ #
    # Station 1-A: PDF ingestion
    # ------------------------------------------------------------------ #
    for path in pdf_paths:
        label = str(path)
        logger.info("Pipeline → PDF: %s", label)
        chunks = _safe_ingest(label, ingest_pdf, path)
        if not chunks and Path(path).exists():
            # File existed but yielded nothing — record as a soft failure.
            failed_sources.append(label)
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ #
    # Station 1-B: JSON / CSV ingestion
    # ------------------------------------------------------------------ #
    for path in json_paths:
        label = str(path)
        logger.info("Pipeline → JSON/CSV: %s", label)
        kwargs = {"template": json_template} if json_template else {}
        chunks = _safe_ingest(label, ingest_json, path, **kwargs)
        if not chunks and Path(path).exists():
            failed_sources.append(label)
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ #
    # Station 1-C: Markdown ingestion
    # ------------------------------------------------------------------ #
    for path in markdown_paths:
        label = str(path)
        logger.info("Pipeline → Markdown: %s", label)
        chunks = _safe_ingest(label, ingest_markdown, path)
        if not chunks and Path(path).exists():
            failed_sources.append(label)
        all_chunks.extend(chunks)

    # ------------------------------------------------------------------ #
    # Deduplication pass
    # ------------------------------------------------------------------ #
    total_raw = len(all_chunks)
    logger.info("Total raw chunks before deduplication: %d", total_raw)

    unique_chunks, duplicates_removed = _deduplicate(all_chunks)

    if duplicates_removed:
        logger.info(
            "Deduplication removed %d duplicate chunk(s). "
            "%d unique chunk(s) remain.",
            duplicates_removed,
            len(unique_chunks),
        )

    if failed_sources:
        logger.warning(
            "%d source(s) failed or yielded no content: %s",
            len(failed_sources),
            failed_sources,
        )

    logger.info(
        "Pipeline complete. %d chunk(s) ready for embedding.",
        len(unique_chunks),
    )

    return IngestionResult(
        chunks=unique_chunks,
        total_raw=total_raw,
        duplicates_removed=duplicates_removed,
        failed_sources=failed_sources,
    )