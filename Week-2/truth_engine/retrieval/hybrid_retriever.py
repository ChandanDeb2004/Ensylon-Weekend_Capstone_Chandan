# RRF fusion of semantic + BM25
"""
retrieval/hybrid_retriever.py

Merges results from the vector store (semantic) and BM25 store (keyword)
using Reciprocal Rank Fusion (RRF) to produce a single unified ranking.

RRF sidesteps the incompatible score scales of the two retrieval systems
by operating purely on rank positions, not raw scores.  A chunk ranked
highly by both systems receives a strong combined RRF signal.

Reference:
    Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods."
"""

from __future__ import annotations

import logging
from typing import Any

from storage.bm25_store import BM25Store
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard RRF smoothing constant from Cormack et al. (2009).
# Dampens the advantage of rank-1 over rank-2, preventing top results
# from dominating the fusion score disproportionately.
DEFAULT_RRF_K: int = 60

# Number of candidates fetched from each store before fusion.
# Should be larger than the final top_k requested by the caller.
DEFAULT_CANDIDATES_PER_STORE: int = 20


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RetrievalResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Dual-index retriever using Reciprocal Rank Fusion.

    Queries both a :class:`~storage.VectorStore` and a
    :class:`~storage.BM25Store` simultaneously, then merges the two
    ranked result lists into a single unified ranking via RRF.

    Both stores are injected at construction time, keeping this class
    decoupled from storage configuration and trivially testable with
    mock stores.

    Attributes:
        rrf_k: The RRF smoothing constant in use.
        candidates_per_store: Candidates fetched from each store per query.

    Example:
        >>> retriever = HybridRetriever(vector_store, bm25_store)
        >>> results = retriever.retrieve(
        ...     query="how do I restart the daemon?",
        ...     top_k=10,
        ...     source_tiers=[1, 2],
        ... )
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        rrf_k: int = DEFAULT_RRF_K,
        candidates_per_store: int = DEFAULT_CANDIDATES_PER_STORE,
    ) -> None:
        """Initialise the retriever with injected store dependencies.

        Args:
            vector_store: An initialised :class:`~storage.VectorStore`
                instance for semantic retrieval.
            bm25_store: An initialised :class:`~storage.BM25Store`
                instance for keyword retrieval.
            rrf_k: RRF smoothing constant.  The default of 60 is the
                empirically validated value from the original paper and
                should rarely need changing.
            candidates_per_store: How many results to fetch from each
                store before fusion.  Must be ≥ the ``top_k`` value
                passed to :meth:`retrieve`.  Higher values give RRF
                more signal but increase fusion cost slightly.

        Raises:
            ValueError: If ``rrf_k`` is not a positive integer, or if
                ``candidates_per_store`` is less than 1.
        """
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}.")
        if candidates_per_store < 1:
            raise ValueError(
                f"candidates_per_store must be ≥ 1, got {candidates_per_store}."
            )

        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self.rrf_k = rrf_k
        self.candidates_per_store = candidates_per_store

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_score(rank: int, k: int) -> float:
        """Compute the RRF contribution for a single rank position.

        Args:
            rank: 1-based rank position of a document in a result list.
            k: RRF smoothing constant.

        Returns:
            The scalar RRF contribution: ``1.0 / (k + rank)``.
        """
        return 1.0 / (k + rank)

    def _fuse(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge two ranked result lists into one via Reciprocal Rank Fusion.

        Each unique chunk (identified by its text content) accumulates RRF
        score contributions from every list in which it appears.  Chunks
        present in both lists naturally accumulate higher scores than those
        appearing in only one.

        The ``rrf_score`` and ``sources`` keys are added to each result
        dict so that downstream components can inspect the fusion signal.

        Args:
            vector_results: Ordered list of results from the vector store,
                most similar first.
            bm25_results: Ordered list of results from the BM25 store,
                highest-scoring first.

        Returns:
            A new list of result dicts sorted by descending RRF score.
            Each dict contains all original keys from the source result
            plus two new keys:

            - ``"rrf_score"`` (float): Combined RRF score.
            - ``"sources"`` (list[str]): Which stores contributed,
              e.g. ``["vector", "bm25"]``.
        """
        # Map chunk text → accumulated RRF data.
        # Using text as the key is safe because the pipeline deduplicates
        # chunks before storage, so identical text == identical chunk.
        fused: dict[str, dict[str, Any]] = {}

        ranked_lists: list[tuple[str, list[RetrievalResult]]] = [
            ("vector", vector_results),
            ("bm25", bm25_results),
        ]

        for source_name, results in ranked_lists:
            for rank, result in enumerate(results, start=1):
                text = result["text"]
                score = self._rrf_score(rank, self.rrf_k)

                if text not in fused:
                    # First time seeing this chunk — initialise entry.
                    fused[text] = {
                        **result,
                        "rrf_score": 0.0,
                        "sources": [],
                    }

                fused[text]["rrf_score"] += score
                if source_name not in fused[text]["sources"]:
                    fused[text]["sources"].append(source_name)

        merged = list(fused.values())
        merged.sort(key=lambda x: x["rrf_score"], reverse=True)

        logger.debug(
            "RRF fusion: %d vector + %d BM25 → %d unique candidates.",
            len(vector_results),
            len(bm25_results),
            len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        source_tiers: list[int] | None = None,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval and return the top-k RRF-ranked chunks.

        Queries both the vector store and the BM25 store in sequence,
        fuses the results via RRF, and returns the top ``top_k`` chunks
        from the combined ranking.

        If one store returns no results (e.g. the BM25 index is empty),
        the fusion degrades gracefully to single-source ranking.

        Args:
            query: The natural-language or technical query string.
            top_k: Number of results to return after fusion.  Must be ≤
                ``candidates_per_store``.
            source_tiers: Optional list of source tier integers to filter
                by.  Passed through to both stores unchanged.  ``None``
                disables filtering.

        Returns:
            A list of up to ``top_k`` result dicts, ordered by descending
            RRF score.  Each dict contains:

            - ``"text"`` (str): Chunk content.
            - ``"metadata"`` (dict): Stored metadata.
            - ``"rrf_score"`` (float): Combined RRF score.
            - ``"sources"`` (list[str]): Contributing stores.
            - ``"distance"`` (float, optional): From vector store.
            - ``"score"`` (float, optional): From BM25 store.
            - ``"id"`` (str, optional): Content MD5 from vector store.

        Raises:
            ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k < 1:
            raise ValueError(f"top_k must be ≥ 1, got {top_k}.")

        candidates = self.candidates_per_store
        if top_k > candidates:
            logger.warning(
                "top_k (%d) exceeds candidates_per_store (%d). "
                "Raising candidates_per_store to match.",
                top_k,
                candidates,
            )
            candidates = top_k

        logger.info(
            "Hybrid retrieve: query='%s …' top_k=%d tiers=%s",
            query[:60],
            top_k,
            source_tiers,
        )

        # ---------------------------------------------------------------- #
        # Query both stores.
        # ---------------------------------------------------------------- #
        vector_results = self._vector_store.query(
            query_text=query,
            top_k=candidates,
            source_tiers=source_tiers,
        )
        logger.debug("Vector store returned %d result(s).", len(vector_results))

        bm25_results = self._bm25_store.query(
            query_text=query,
            top_k=candidates,
            source_tiers=source_tiers,
        )
        logger.debug("BM25 store returned %d result(s).", len(bm25_results))

        if not vector_results and not bm25_results:
            logger.warning("Both stores returned no results for query: '%s'", query)
            return []

        # ---------------------------------------------------------------- #
        # Fuse and truncate.
        # ---------------------------------------------------------------- #
        fused = self._fuse(vector_results, bm25_results)
        top_results = fused[:top_k]

        logger.info(
            "Hybrid retrieval complete: returning %d / %d fused candidate(s).",
            len(top_results),
            len(fused),
        )
        return top_results