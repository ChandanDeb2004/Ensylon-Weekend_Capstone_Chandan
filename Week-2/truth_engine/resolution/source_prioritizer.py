# Applies source tier ranking rules
"""
resolution/source_prioritizer.py

Applies deterministic source-tier priority rules to resolve conflicts.

No LLM is involved — priority decisions are fully deterministic and
auditable.  Lower source_tier number always wins.  Suppressed chunks
are annotated and preserved in the resolution log rather than deleted,
enabling downstream citation of overridden sources.

Tier definitions (from config):
    Tier 1 — Primary source (e.g. official PDF manual).     Highest authority.
    Tier 2 — Secondary source (e.g. incident log CSV).      Medium authority.
    Tier 3 — Tertiary source (e.g. wiki Markdown).          Lowest authority.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — these should be imported from config.py in production.
# Defined here as module-level constants for self-contained clarity.
# ---------------------------------------------------------------------------

# Confidence penalty applied multiplicatively to surviving chunks by tier.
# Tier 1 chunks are authoritative — no penalty.
TIER_PENALTIES: dict[int, float] = {
    1: 0.00,   # No penalty — authoritative source.
    2: 0.10,   # 10% confidence reduction — secondary source.
    3: 0.20,   # 20% confidence reduction — tertiary source.
}

# Default confidence for any chunk that has not been scored upstream.
DEFAULT_CONFIDENCE: float = 1.0



from config import TIER_PENALTIES, DEFAULT_TIER_PENALTY
# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Chunk = dict[str, Any]
ConflictRecord = dict[str, Any]
PrioritizationResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tier(chunk: Chunk) -> int:
    """Extract the source tier from a chunk's metadata.

    Args:
        chunk: A chunk dict with a ``"metadata"`` key.

    Returns:
        The integer ``source_tier`` value, or ``99`` as a safe fallback
        for chunks with missing or malformed metadata.
    """
    return chunk.get("metadata", {}).get("source_tier", 99)


def _apply_confidence_penalty(
    chunk: Chunk,
    penalty: float,
) -> Chunk:
    """Return a new chunk dict with a confidence penalty applied.

    The ``confidence`` value is taken from the chunk if present,
    otherwise ``DEFAULT_CONFIDENCE`` is used.  The penalty is applied
    multiplicatively: ``new_confidence = confidence * (1 - penalty)``.

    The original chunk is never mutated — a new dict is returned.

    Args:
        chunk: The source chunk dict.
        penalty: Fractional penalty in [0.0, 1.0].  A penalty of 0.10
            reduces confidence by 10%.

    Returns:
        A new chunk dict with updated ``"confidence"`` key.
    """
    original_confidence: float = chunk.get("confidence", DEFAULT_CONFIDENCE)
    new_confidence: float = round(original_confidence * (1.0 - penalty), 4)
    return {**chunk, "confidence": new_confidence}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SourcePrioritizer:
    """Deterministic conflict resolver based on source tier hierarchy.

    For each conflict, the chunk from the lower-numbered tier (higher
    authority) is retained.  The losing chunk is annotated with
    suppression metadata and moved to the resolution log.

    Surviving chunks from tiers 2 and 3 receive a confidence penalty
    proportional to their tier level, reflecting their lower authority
    relative to tier-1 sources.

    Example:
        >>> prioritizer = SourcePrioritizer()
        >>> result = prioritizer.resolve(
        ...     chunks=retrieved_chunks,
        ...     conflicts=detected_conflicts,
        ... )
        >>> final_chunks = result["final_chunks"]
        >>> suppressed   = result["suppressed_chunks"]
    """

    def __init__(
        self,
        tier_penalties: dict[int, float] | None = None,
    ) -> None:
        """Initialise the prioritizer with optional custom tier penalties.

        Args:
            tier_penalties: Optional dict mapping tier integers to
                fractional confidence penalty values.  Overrides
                :data:`TIER_PENALTIES` when provided.
                Example: ``{1: 0.0, 2: 0.15, 3: 0.25}``.
        """
        self._penalties: dict[int, float] = (
            tier_penalties if tier_penalties is not None
            else TIER_PENALTIES
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _suppress(
        self,
        chunk: Chunk,
        winner_source: str,
        winner_tier: int,
    ) -> Chunk:
        """Annotate a chunk as suppressed without deleting it.

        Returns a new dict — the original chunk is never mutated.

        Args:
            chunk: The losing chunk to suppress.
            winner_source: Source name of the winning chunk.
            winner_tier: Source tier of the winning chunk.

        Returns:
            A new chunk dict with suppression annotation keys added:

            - ``"suppressed"`` (bool): Always ``True``.
            - ``"suppression_reason"`` (str): Human-readable reason.
            - ``"overridden_by_source"`` (str): Winning source name.
            - ``"overridden_by_tier"`` (int): Winning source tier.
        """
        loser_tier = _get_tier(chunk)
        loser_source = chunk.get("metadata", {}).get("source_name", "unknown")
        return {
            **chunk,
            "suppressed": True,
            "suppression_reason": (
                f"outranked_by_tier_{winner_tier}"
            ),
            "overridden_by_source": winner_source,
            "overridden_by_tier": winner_tier,
            "original_source": loser_source,
            "original_tier": loser_tier,
        }

    def _resolve_conflict(
        self,
        conflict: ConflictRecord,
    ) -> tuple[Chunk, Chunk]:
        """Determine the winner and loser for a single conflict record.

        Applies the deterministic tier rule: lower tier number wins.
        In case of a tier tie (same tier, different source), chunk_a is
        retained by convention and chunk_b is suppressed.

        Args:
            conflict: A conflict record dict as produced by
                :class:`~resolution.ConflictDetector`.

        Returns:
            A 2-tuple of ``(winner_chunk, suppressed_chunk)``.
        """
        chunk_a = conflict["chunk_a"]
        chunk_b = conflict["chunk_b"]
        tier_a = conflict["tier_a"]
        tier_b = conflict["tier_b"]
        source_a = conflict["source_a"]
        source_b = conflict["source_b"]

        if tier_a <= tier_b:
            winner, loser = chunk_a, chunk_b
            winner_source, winner_tier = source_a, tier_a
        else:
            winner, loser = chunk_b, chunk_a
            winner_source, winner_tier = source_b, tier_b

        suppressed = self._suppress(loser, winner_source, winner_tier)

        logger.info(
            "Conflict resolved: '%s' (tier %d) WINS over '%s' (tier %d).",
            winner_source,
            winner_tier,
            loser.get("metadata", {}).get("source_name", "unknown"),
            _get_tier(loser),
        )

        return winner, suppressed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        chunks: list[Chunk],
        conflicts: list[ConflictRecord],
    ) -> PrioritizationResult:
        """Apply tier-based priority rules to all detected conflicts.

        Iterates over each conflict record, suppresses the lower-priority
        chunk, then applies confidence penalties to all surviving chunks
        based on their source tier.

        Args:
            chunks: The full list of chunks under consideration.
            conflicts: List of conflict records from
                :class:`~resolution.ConflictDetector`.

        Returns:
            A dict containing:

            - ``"final_chunks"`` (list[dict]): Surviving chunks with
              confidence penalties applied.
            - ``"suppressed_chunks"`` (list[dict]): Annotated losing
              chunks, preserved for citation and reporting.
            - ``"resolution_log"`` (list[str]): Human-readable log
              entries, one per resolved conflict.

            If ``conflicts`` is empty, ``final_chunks`` equals the
            input ``chunks`` (with penalties applied) and the other
            lists are empty.
        """
        suppressed_texts: set[str] = set()
        suppressed_chunks: list[Chunk] = []
        resolution_log: list[str] = []

        # ---------------------------------------------------------------- #
        # Resolve each conflict and collect suppressed chunk texts.
        # ---------------------------------------------------------------- #
        for conflict in conflicts:
            winner, suppressed = self._resolve_conflict(conflict)

            suppressed_text = suppressed["text"]
            if suppressed_text not in suppressed_texts:
                suppressed_texts.add(suppressed_text)
                suppressed_chunks.append(suppressed)

                log_entry = (
                    f"CONFLICT RESOLVED — "
                    f"'{suppressed['overridden_by_source']}' "
                    f"(tier {suppressed['overridden_by_tier']}) "
                    f"overrides "
                    f"'{suppressed['original_source']}' "
                    f"(tier {suppressed['original_tier']}). "
                    f"Reason: {suppressed['suppression_reason']}. "
                    f"Similarity: {conflict['similarity']:.4f}."
                )
                resolution_log.append(log_entry)
                logger.info(log_entry)

        # ---------------------------------------------------------------- #
        # Build final chunk list: exclude suppressed, apply tier penalties.
        # ---------------------------------------------------------------- #
        final_chunks: list[Chunk] = []
        for chunk in chunks:
            if chunk["text"] in suppressed_texts:
                continue

            tier = _get_tier(chunk)
            penalty = self._penalties.get(tier, DEFAULT_TIER_PENALTY)
            penalised = _apply_confidence_penalty(chunk, penalty)
            final_chunks.append(penalised)

        logger.info(
            "Prioritization complete: %d final chunk(s), "
            "%d suppressed, %d conflict(s) resolved.",
            len(final_chunks),
            len(suppressed_chunks),
            len(conflicts),
        )

        return {
            "final_chunks": final_chunks,
            "suppressed_chunks": suppressed_chunks,
            "resolution_log": resolution_log,
        }