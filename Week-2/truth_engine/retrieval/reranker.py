# Cross-encoder re-ranking logic
"""
retrieval/reranker.py

Applies a Cross-Encoder model to re-score and re-rank the top candidates
from hybrid retrieval.

A cross-encoder reads each (query, chunk) pair jointly as a single input,
giving it far greater accuracy than a bi-encoder at the cost of speed.
This two-stage design (fast bi-encoder over all chunks → precise
cross-encoder over top-20) is the standard production pattern for RAG
pipelines: the cross-encoder never sees more than a small candidate set.

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
    — Trained on MS MARCO passage ranking.
    — Runs on CPU in ~100–300ms for 20 candidates.
    — Well-calibrated for technical Q&A relevance.
"""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RankedResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Reranker:
    """Cross-encoder re-ranker for RAG candidate refinement.

    Takes the top-N candidates from :class:`~retrieval.HybridRetriever`
    and scores each ``(query, chunk)`` pair jointly using a Cross-Encoder
    model, returning only the top-k most relevant chunks.

    The cross-encoder scores are logits — meaningful only in relative
    terms within a single query's result set, not as absolute thresholds.

    Attributes:
        model_name: HuggingFace model identifier of the cross-encoder.

    Example:
        >>> reranker = Reranker()
        >>> top_chunks = reranker.rerank(
        ...     query="how long does Procedure X take?",
        ...     candidates=hybrid_results,   # list[dict] from HybridRetriever
        ...     top_k=5,
        ... )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        max_length: int = 512,
    ) -> None:
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace model identifier.  Must be a
                cross-encoder model, not a bi-encoder.  The default
                ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is recommended
                for CPU inference over small candidate sets.
            max_length: Maximum token length for each ``(query, chunk)``
                pair.  Pairs exceeding this length are truncated.
                512 is the safe default for MiniLM-based models.

        Raises:
            RuntimeError: If the model cannot be loaded from HuggingFace
                or the local cache.
        """
        self.model_name = model_name
        logger.info("Loading cross-encoder model '%s' …", model_name)

        try:
            self._model: CrossEncoder = CrossEncoder(
                model_name,
                max_length=max_length,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}': {exc}"
            ) from exc

        logger.info("Cross-encoder model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RankedResult]:
        """Re-score and re-rank a candidate list using the cross-encoder.

        Constructs ``(query, chunk_text)`` pairs for every candidate and
        scores them in a single batched ``predict`` call.  Results are
        sorted by descending cross-encoder score and the top ``top_k``
        are returned.

        Each result dict is augmented with a ``"cross_encoder_score"``
        key containing the raw logit score for transparency and
        downstream debugging.

        Args:
            query: The original query string, identical to the one passed
                to :class:`~retrieval.HybridRetriever`.
            candidates: List of candidate dicts as returned by
                :meth:`~retrieval.HybridRetriever.retrieve`.  Each dict
                must contain at minimum a ``"text"`` key.
            top_k: Number of top-ranked results to return.  If
                ``top_k`` exceeds ``len(candidates)``, all candidates
                are returned (sorted by cross-encoder score).

        Returns:
            A list of up to ``top_k`` result dicts, sorted by descending
            ``"cross_encoder_score"``.  Each dict contains all original
            keys from the candidate plus:

            - ``"cross_encoder_score"`` (float): Raw logit score.
            - ``"cross_encoder_rank"`` (int): 1-based rank after
              re-scoring.

            Returns an empty list if ``candidates`` is empty.

        Raises:
            ValueError: If ``query`` is empty or ``top_k`` is less than 1.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k < 1:
            raise ValueError(f"top_k must be ≥ 1, got {top_k}.")

        if not candidates:
            logger.warning("rerank called with empty candidate list.")
            return []

        effective_top_k = min(top_k, len(candidates))

        logger.info(
            "Re-ranking %d candidate(s) → top %d  |  query='%s …'",
            len(candidates),
            effective_top_k,
            query[:60],
        )

        # ---------------------------------------------------------------- #
        # Build (query, chunk) pairs and score in one batched call.
        # CrossEncoder.predict is significantly faster called once over a
        # list than called N times individually.
        # ---------------------------------------------------------------- #
        pairs: list[tuple[str, str]] = [
            (query, candidate["text"]) for candidate in candidates
        ]

        scores: list[float] = self._model.predict(pairs).tolist()

        # ---------------------------------------------------------------- #
        # Annotate each candidate with its cross-encoder score.
        # ---------------------------------------------------------------- #
        scored: list[RankedResult] = []
        for candidate, score in zip(candidates, scores):
            scored.append(
                {
                    **candidate,
                    "cross_encoder_score": score,
                }
            )

        # ---------------------------------------------------------------- #
        # Sort by descending score and assign final ranks.
        # ---------------------------------------------------------------- #
        scored.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        results: list[RankedResult] = []
        for rank, result in enumerate(scored[:effective_top_k], start=1):
            results.append({**result, "cross_encoder_rank": rank})

        logger.info(
            "Re-ranking complete. Top score: %.4f  |  Bottom score: %.4f",
            results[0]["cross_encoder_score"],
            results[-1]["cross_encoder_score"],
        )

        return results