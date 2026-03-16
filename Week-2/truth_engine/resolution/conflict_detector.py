# Detects contradictions between chunks
"""
resolution/conflict_detector.py

Detects contradictions between chunks from different source documents.

Uses a two-stage approach to keep LLM API costs proportional to actual
conflict density rather than O(n²) in chunk count:

  Stage 1 — Cosine gate: compute pairwise cosine similarity between
            cross-source chunk embeddings.  Only pairs above the
            similarity threshold (default 0.85) are candidates —
            they are topically related enough that differences between
            them constitute contradictions rather than unrelated facts.

  Stage 2 — LLM binary classification: for each candidate pair, make a
            single targeted API call with a strict binary prompt.
            The model replies only CONFLICT or NO_CONFLICT.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np

from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Chunk = dict[str, Any]
ConflictRecord = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        vec_a: First embedding vector as a list of floats.
        vec_b: Second embedding vector as a list of floats.

    Returns:
        Cosine similarity in the range [-1.0, 1.0].  Returns 0.0 if
        either vector has zero magnitude to avoid division by zero.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_conflict_prompt(text_a: str, text_b: str) -> str:
    """Build the binary classification prompt for LLM conflict detection.

    The prompt is deliberately terse and binary.  Free-text explanations
    are explicitly forbidden to prevent response parsing complexity.

    Args:
        text_a: Text content of the first chunk.
        text_b: Text content of the second chunk.

    Returns:
        A formatted prompt string ready for the LLM.
    """
    return (
        "You are a fact-conflict detector. Your only job is to determine "
        "whether two passages state DIFFERENT values for the SAME fact.\n\n"
        "Passage A:\n"
        f"{text_a.strip()}\n\n"
        "Passage B:\n"
        f"{text_b.strip()}\n\n"
        "Do these two passages state different values for the same fact? "
        "Reply with exactly one word — either CONFLICT or NO_CONFLICT. "
        "Do not explain. Do not hedge. Output nothing else."
    )


def _parse_conflict_response(response: str) -> bool:
    """Parse the LLM binary response into a boolean conflict flag.

    Normalises the response to handle minor LLM non-compliance such as
    punctuation, mixed case, or brief prefixes.  Defaults to ``False``
    (no conflict) when the response is ambiguous — false negatives are
    safer than false positives in a conflict detection system.

    Args:
        response: Raw string returned by the LLM.

    Returns:
        ``True`` if the response indicates a conflict, ``False`` otherwise.
    """
    normalised = response.strip().upper().rstrip(".")
    if "NO_CONFLICT" in normalised:
        return False
    if "CONFLICT" in normalised:
        return True
    logger.warning(
        "Ambiguous LLM conflict response '%s'. Defaulting to NO_CONFLICT.",
        response[:80],
    )
    return False


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ConflictDetector:
    """Two-stage cross-source conflict detector.

    Stage 1 uses cosine similarity to gate candidate pairs efficiently.
    Stage 2 uses a binary LLM prompt to classify gated pairs.

    The LLM callable is injected at construction time, keeping this class
    decoupled from any specific LLM client (OpenAI, Anthropic, local, etc.).

    Attributes:
        similarity_threshold: Cosine similarity above which two chunks
            are considered topically related and worth classifying.

    Example:
        >>> detector = ConflictDetector(
        ...     vector_store=vector_store,
        ...     llm_callable=my_llm_fn,
        ...     similarity_threshold=0.85,
        ... )
        >>> conflicts = detector.detect(chunks)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_callable: callable,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialise the conflict detector.

        Args:
            vector_store: An initialised :class:`~storage.VectorStore`
                instance.  Used to retrieve stored embeddings so that
                chunks are not re-embedded from scratch.
            llm_callable: A callable with signature
                ``(prompt: str) -> str`` that sends a prompt to an LLM
                and returns the raw text response.  The caller is
                responsible for authentication, rate limiting, and retry
                logic.
            similarity_threshold: Cosine similarity threshold for the
                Stage 1 gate.  Pairs below this value are assumed to be
                about different topics and are never sent to the LLM.
                Must be in the range (0.0, 1.0].

        Raises:
            ValueError: If ``similarity_threshold`` is outside (0.0, 1.0].
        """
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0.0, 1.0], "
                f"got {similarity_threshold}."
            )

        self._vector_store = vector_store
        self._llm = llm_callable
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_embeddings(
        self, chunks: list[Chunk]
    ) -> dict[str, list[float]]:
        """Retrieve stored embeddings for a list of chunks from ChromaDB.

        Uses the content-derived MD5 IDs to fetch embeddings in a single
        batch call rather than one call per chunk.  Chunks whose IDs are
        not found in the store are logged and skipped.

        Args:
            chunks: List of chunk dicts, each containing a ``"text"``
                key used to derive the ChromaDB document ID.

        Returns:
            A dict mapping chunk text → embedding vector (list of floats).
        """
        import hashlib

        id_to_text: dict[str, str] = {
            hashlib.md5(c["text"].strip().encode("utf-8")).hexdigest(): c["text"]
            for c in chunks
        }

        if not id_to_text:
            return {}

        raw = self._vector_store._collection.get(
            ids=list(id_to_text.keys()),
            include=["embeddings"],
        )

        text_to_embedding: dict[str, list[float]] = {}
        for doc_id, embedding in zip(raw["ids"], raw["embeddings"]):
            text = id_to_text.get(doc_id)
            if text is not None:
                text_to_embedding[text] = embedding

        missing = len(chunks) - len(text_to_embedding)
        if missing > 0:
            logger.warning(
                "%d chunk(s) had no stored embedding and will be skipped "
                "during conflict detection.",
                missing,
            )

        return text_to_embedding

    def _get_cross_source_pairs(
        self, chunks: list[Chunk]
    ) -> list[tuple[Chunk, Chunk]]:
        """Generate all cross-source chunk pairs for comparison.

        Only pairs where the two chunks originate from different source
        documents are included.  Chunks from the same source cannot
        conflict with each other by definition within this system.

        Args:
            chunks: Full list of chunks, each with a ``"metadata"`` dict
                containing a ``"source_name"`` key.

        Returns:
            A list of 2-tuples, each containing two chunks from different
            sources.
        """
        pairs: list[tuple[Chunk, Chunk]] = []
        for chunk_a, chunk_b in combinations(chunks, 2):
            source_a = chunk_a.get("metadata", {}).get("source_name", "")
            source_b = chunk_b.get("metadata", {}).get("source_name", "")
            if source_a != source_b:
                pairs.append((chunk_a, chunk_b))
        return pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, chunks: list[Chunk]) -> list[ConflictRecord]:
        """Run two-stage conflict detection over a list of chunks.

        Stage 1 computes cosine similarity for all cross-source pairs and
        discards pairs below ``similarity_threshold``.  Stage 2 sends
        each remaining candidate pair to the LLM for binary classification.

        Args:
            chunks: List of chunk dicts as returned by
                :class:`~retrieval.Reranker` or assembled upstream.
                Each dict must contain ``"text"`` and ``"metadata"``
                (with ``"source_name"`` and ``"source_tier"`` keys).

        Returns:
            A list of conflict record dicts.  Each record contains:

            - ``"chunk_a"`` (dict): First conflicting chunk.
            - ``"chunk_b"`` (dict): Second conflicting chunk.
            - ``"similarity"`` (float): Cosine similarity score.
            - ``"source_a"`` (str): Source name of chunk A.
            - ``"source_b"`` (str): Source name of chunk B.
            - ``"tier_a"`` (int): Source tier of chunk A.
            - ``"tier_b"`` (int): Source tier of chunk B.

            Returns an empty list if no conflicts are detected or if
            ``chunks`` contains fewer than two elements.
        """
        if len(chunks) < 2:
            logger.info("Fewer than 2 chunks — conflict detection skipped.")
            return []

        logger.info(
            "Starting conflict detection on %d chunk(s).", len(chunks)
        )

        # ---------------------------------------------------------------- #
        # Fetch stored embeddings in a single batch call.
        # ---------------------------------------------------------------- #
        text_to_embedding = self._fetch_embeddings(chunks)

        cross_source_pairs = self._get_cross_source_pairs(chunks)
        logger.info(
            "%d cross-source pair(s) to evaluate.", len(cross_source_pairs)
        )

        # ---------------------------------------------------------------- #
        # Stage 1: Cosine similarity gate.
        # ---------------------------------------------------------------- #
        candidates: list[tuple[Chunk, Chunk, float]] = []
        for chunk_a, chunk_b in cross_source_pairs:
            emb_a = text_to_embedding.get(chunk_a["text"])
            emb_b = text_to_embedding.get(chunk_b["text"])

            if emb_a is None or emb_b is None:
                logger.debug(
                    "Skipping pair — embedding missing for one or both chunks."
                )
                continue

            similarity = _cosine_similarity(emb_a, emb_b)
            if similarity >= self.similarity_threshold:
                candidates.append((chunk_a, chunk_b, similarity))

        logger.info(
            "Stage 1 (cosine gate): %d / %d pair(s) above threshold %.2f.",
            len(candidates),
            len(cross_source_pairs),
            self.similarity_threshold,
        )

        # ---------------------------------------------------------------- #
        # Stage 2: LLM binary classification.
        # ---------------------------------------------------------------- #
        conflicts: list[ConflictRecord] = []
        for chunk_a, chunk_b, similarity in candidates:
            prompt = _build_conflict_prompt(chunk_a["text"], chunk_b["text"])

            try:
                response = self._llm(prompt)
                is_conflict = _parse_conflict_response(response)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "LLM call failed during conflict classification: %s. "
                    "Treating pair as NO_CONFLICT.",
                    exc,
                )
                is_conflict = False

            if is_conflict:
                meta_a = chunk_a.get("metadata", {})
                meta_b = chunk_b.get("metadata", {})
                record: ConflictRecord = {
                    "chunk_a": chunk_a,
                    "chunk_b": chunk_b,
                    "similarity": round(similarity, 4),
                    "source_a": meta_a.get("source_name", "unknown"),
                    "source_b": meta_b.get("source_name", "unknown"),
                    "tier_a": meta_a.get("source_tier", 99),
                    "tier_b": meta_b.get("source_tier", 99),
                }
                conflicts.append(record)
                logger.info(
                    "CONFLICT detected: '%s' (tier %d) vs '%s' (tier %d)  "
                    "similarity=%.4f",
                    record["source_a"],
                    record["tier_a"],
                    record["source_b"],
                    record["tier_b"],
                    similarity,
                )

        logger.info(
            "Conflict detection complete: %d conflict(s) found.", len(conflicts)
        )
        return conflicts