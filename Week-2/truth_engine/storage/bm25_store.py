# BM25 index build & keyword search
"""
storage/bm25_store.py

Manages a BM25 keyword index for exact-term retrieval.

Complements the vector store by excelling where semantic search fails:
exact technical strings, error codes, and precise terminology.
The index is persisted to disk as a JSON corpus (texts + metadata)
and rebuilt on load, avoiding pickle version compatibility issues.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STORE_PATH: Path = Path("./bm25_store")
CORPUS_FILENAME: str = "corpus.json"

# Tokenisation pattern: split on whitespace and punctuation, but preserve
# tokens like ERR_CODE_0x4F2 and version strings (e.g. v1.2.3) intact.
_TOKEN_PATTERN: re.Pattern = re.compile(r"[^\w]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and querying.

    Splits on non-word characters (whitespace, punctuation) while
    preserving technical tokens such as error codes, hex values, and
    version strings.  Empty tokens produced by consecutive delimiters
    are discarded.

    Deliberately does NOT lowercase, because exact-case matching is
    critical for technical identifiers like ``ERR_CODE_0x4F2``.

    Args:
        text: Raw input string to tokenize.

    Returns:
        A list of non-empty string tokens.

    Example:
        >>> _tokenize("Error code ERR_CODE_0x4F2 on restart.")
        ['Error', 'code', 'ERR_CODE_0x4F2', 'on', 'restart']
    """
    tokens = _TOKEN_PATTERN.split(text)
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class BM25Store:
    """Keyword-based BM25 index for exact-term and technical string retrieval.

    Wraps :class:`~rank_bm25.BM25Okapi` with a persistent JSON corpus
    and a clean query interface that mirrors :class:`~storage.VectorStore`.

    The BM25 index itself is not serialised; instead the raw corpus is
    saved to disk and the index is rebuilt on load.  This avoids
    ``pickle`` version incompatibilities across Python upgrades.

    Attributes:
        store_path: Directory where the corpus JSON file is persisted.

    Example:
        >>> store = BM25Store(store_path="./bm25_store")
        >>> store.add_chunks(
        ...     chunks=["Daemon crashed with ERR_CODE_0x4F2 on startup."],
        ...     metadatas=[{
        ...         "source_tier": 1,
        ...         "source_name": "incident_log.csv",
        ...         "page_number": 0,
        ...         "section": "",
        ...     }],
        ... )
        >>> results = store.query("ERR_CODE_0x4F2", top_k=3)
    """

    def __init__(self, store_path: str | Path = DEFAULT_STORE_PATH) -> None:
        """Initialise the BM25 store, loading an existing corpus if present.

        Args:
            store_path: Directory path for corpus persistence.  Created
                automatically if it does not exist.
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Internal state ------------------------------------------------
        self._corpus_texts: list[str] = []
        self._corpus_metadatas: list[dict[str, Any]] = []
        self._index: BM25Okapi | None = None

        corpus_file = self._corpus_path()
        if corpus_file.exists():
            self._load(corpus_file)
        else:
            logger.info(
                "No existing BM25 corpus found at '%s'. "
                "Starting with empty index.",
                corpus_file,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _corpus_path(self) -> Path:
        """Return the absolute path to the corpus JSON file."""
        return self.store_path / CORPUS_FILENAME

    def _rebuild_index(self) -> None:
        """(Re)build the BM25Okapi index from the current corpus texts.

        Called after every ``add_chunks`` and after ``_load``.
        Guards against an empty corpus to prevent internal BM25
        division-by-zero errors.
        """
        if not self._corpus_texts:
            self._index = None
            return

        tokenized_corpus = [_tokenize(text) for text in self._corpus_texts]
        self._index = BM25Okapi(tokenized_corpus)
        logger.debug(
            "BM25 index rebuilt over %d document(s).", len(self._corpus_texts)
        )

    def _save(self) -> None:
        """Persist the corpus texts and metadata to disk as JSON.

        The BM25 index is intentionally not saved; it is rebuilt from
        the corpus on load.
        """
        payload = {
            "texts": self._corpus_texts,
            "metadatas": self._corpus_metadatas,
        }
        corpus_file = self._corpus_path()
        try:
            corpus_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "BM25 corpus saved: %d document(s) → '%s'.",
                len(self._corpus_texts),
                corpus_file,
            )
        except OSError as exc:
            logger.error(
                "Failed to save BM25 corpus to '%s': %s", corpus_file, exc
            )

    def _load(self, corpus_file: Path) -> None:
        """Load corpus from disk and rebuild the BM25 index.

        Args:
            corpus_file: Path to the ``corpus.json`` file on disk.
        """
        try:
            raw = json.loads(corpus_file.read_text(encoding="utf-8"))
            self._corpus_texts = raw.get("texts", [])
            self._corpus_metadatas = raw.get("metadatas", [])
            self._rebuild_index()
            logger.info(
                "BM25 corpus loaded: %d document(s) from '%s'.",
                len(self._corpus_texts),
                corpus_file,
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to load BM25 corpus from '%s': %s. "
                "Starting with empty index.",
                corpus_file,
                exc,
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a list of text chunks to the BM25 index.

        Duplicate chunks (exact text match) are silently skipped so that
        re-running the pipeline remains idempotent.

        After adding, the BM25 index is rebuilt and the updated corpus
        is persisted to disk.

        Args:
            chunks: List of plain-text chunk strings to index.
            metadatas: Optional list of metadata dicts, one per chunk.
                Must be the same length as ``chunks`` if provided.

        Raises:
            ValueError: If ``metadatas`` length differs from ``chunks``.
        """
        if not chunks:
            logger.warning("add_chunks called with an empty list. No-op.")
            return

        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks but "
                f"{len(metadatas)} metadata dicts."
            )

        resolved_metadatas = metadatas or [{} for _ in chunks]

        # Deduplicate against existing corpus using a set for O(1) lookup.
        existing: set[str] = set(self._corpus_texts)
        added = 0
        for chunk, meta in zip(chunks, resolved_metadatas):
            if chunk.strip() in existing:
                logger.debug("Skipping duplicate chunk (BM25): '%s …'", chunk[:60])
                continue
            self._corpus_texts.append(chunk)
            self._corpus_metadatas.append(meta)
            existing.add(chunk.strip())
            added += 1

        if added == 0:
            logger.info("No new chunks to add to BM25 index.")
            return

        logger.info(
            "Adding %d new chunk(s) to BM25 index. "
            "Total corpus size: %d.",
            added,
            len(self._corpus_texts),
        )

        self._rebuild_index()
        self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_tiers: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k keyword-matched chunks using BM25 scoring.

        Args:
            query_text: The query string.  Technical terms and error
                codes are matched exactly (case-sensitive).
            top_k: Maximum number of results to return.
            source_tiers: Optional whitelist of ``source_tier`` integer
                values.  Chunks whose metadata ``source_tier`` is not
                in this list are excluded before BM25 scoring, mirroring
                the filtering behaviour of :class:`~storage.VectorStore`.

        Returns:
            A list of dicts ordered by descending BM25 score, each
            containing:

            - ``"text"`` (str): The chunk content.
            - ``"metadata"`` (dict): The stored metadata.
            - ``"score"`` (float): The BM25 relevance score.
            - ``"index"`` (int): Position of the chunk in the corpus.

            Returns an empty list if the index is empty or the query
            produces zero positive-scoring results.

        Raises:
            ValueError: If ``query_text`` is empty.
        """
        if not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")

        if self._index is None:
            logger.warning("BM25 index is empty. Returning no results.")
            return []

        query_tokens = _tokenize(query_text)
        all_scores: list[float] = self._index.get_scores(query_tokens).tolist()

        # Build scored candidates, applying source_tier filter if requested.
        candidates: list[dict[str, Any]] = []
        for idx, (score, text, meta) in enumerate(
            zip(all_scores, self._corpus_texts, self._corpus_metadatas)
        ):
            if score <= 0.0:
                continue  # BM25 returns 0.0 for irrelevant documents.

            if source_tiers is not None:
                tier = meta.get("source_tier")
                if tier not in source_tiers:
                    continue

            candidates.append(
                {
                    "text": text,
                    "metadata": meta,
                    "score": score,
                    "index": idx,
                }
            )

        # Sort descending by score; return top_k.
        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = candidates[:top_k]

        logger.debug(
            "BM25 query returned %d result(s) for: '%s'",
            len(results),
            query_text[:80],
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of documents in the corpus.

        Returns:
            Integer count of indexed chunks.
        """
        return len(self._corpus_texts)

    def reset(self) -> None:
        """Wipe the in-memory index and delete the on-disk corpus.

        Intended for testing and pipeline re-runs from scratch.
        """
        logger.warning(
            "Resetting BM25 store at '%s'. All data will be lost.",
            self.store_path,
        )
        self._corpus_texts = []
        self._corpus_metadatas = []
        self._index = None

        corpus_file = self._corpus_path()
        if corpus_file.exists():
            corpus_file.unlink()