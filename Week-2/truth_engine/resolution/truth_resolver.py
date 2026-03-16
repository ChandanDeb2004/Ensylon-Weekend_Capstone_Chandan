# Orchestrates detection + prioritization
"""
resolution/truth_resolver.py

Orchestrates the full truth resolution pipeline for Station 4.

Calls ConflictDetector, passes results to SourcePrioritizer, and
packages everything into a stable ResolutionResult dataclass that
downstream components (prompt builder, citation generator, report
writer) consume without needing to know resolution internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from resolution.conflict_detector import ConflictDetector
from resolution.source_prioritizer import SourcePrioritizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    """Structured output from the truth resolution pipeline.

    This dataclass is the API contract between Station 4 (resolution)
    and all downstream stations (prompt builder, citation generator,
    report writer).  Consumers should depend on this type, never on the
    internal dicts of individual resolution components.

    Attributes:
        final_chunks: Surviving chunks after conflict resolution, with
            confidence penalties applied.  These are the chunks the
            prompt builder will inject into the LLM context.
        conflicts: All conflict records detected by
            :class:`~resolution.ConflictDetector`, regardless of
            resolution outcome.  Preserved for citation generation.
        suppressed_chunks: Annotated chunks that lost conflict resolution.
            Preserved so the system can cite "Source C was overridden
            by Source A on this point."
        resolution_log: Human-readable log strings, one per resolved
            conflict.  Included verbatim in the final report.
        conflict_count: Number of conflicts detected.  Convenience
            attribute — equals ``len(conflicts)``.
        suppressed_count: Number of chunks suppressed.  Convenience
            attribute — equals ``len(suppressed_chunks)``.
    """

    final_chunks: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    suppressed_chunks: list[dict[str, Any]] = field(default_factory=list)
    resolution_log: list[str] = field(default_factory=list)

    @property
    def conflict_count(self) -> int:
        """Number of conflicts detected."""
        return len(self.conflicts)

    @property
    def suppressed_count(self) -> int:
        """Number of chunks suppressed."""
        return len(self.suppressed_chunks)

    def format_log(self) -> str:
        """Return the resolution log as a single human-readable string.

        Returns:
            Newline-joined resolution log entries, or a message
            indicating no conflicts were detected.
        """
        if not self.resolution_log:
            return "No conflicts detected. All sources are consistent."
        return "\n".join(self.resolution_log)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class TruthResolver:
    """Orchestrator for the full truth resolution pipeline.

    Coordinates :class:`~resolution.ConflictDetector` and
    :class:`~resolution.SourcePrioritizer` into a single call that
    returns a :class:`ResolutionResult`.

    Both dependencies are injected at construction time for testability
    and configurability.

    Example:
        >>> resolver = TruthResolver(
        ...     conflict_detector=ConflictDetector(vector_store, llm_fn),
        ...     source_prioritizer=SourcePrioritizer(),
        ... )
        >>> result = resolver.resolve(chunks)
        >>> print(result.format_log())
        >>> final_chunks = result.final_chunks
    """

    def __init__(
        self,
        conflict_detector: ConflictDetector,
        source_prioritizer: SourcePrioritizer,
    ) -> None:
        """Initialise the resolver with injected dependencies.

        Args:
            conflict_detector: An initialised
                :class:`~resolution.ConflictDetector` instance.
            source_prioritizer: An initialised
                :class:`~resolution.SourcePrioritizer` instance.
        """
        self._detector = conflict_detector
        self._prioritizer = source_prioritizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, chunks: list[dict[str, Any]]) -> ResolutionResult:
        """Run the full resolution pipeline on a list of retrieved chunks.

        Executes the two resolution stages in sequence:
        1. :meth:`~resolution.ConflictDetector.detect` — identify
           contradicting cross-source pairs.
        2. :meth:`~resolution.SourcePrioritizer.resolve` — apply tier
           rules, suppress losers, apply confidence penalties.

        Handles the zero-conflict case cleanly: when no conflicts are
        detected, all chunks are returned with confidence penalties
        applied and all log/suppressed lists are empty.

        Args:
            chunks: List of chunk dicts as returned by
                :class:`~retrieval.Reranker`.  Each dict must contain
                ``"text"`` and ``"metadata"`` with ``"source_name"``
                and ``"source_tier"`` keys.

        Returns:
            A :class:`ResolutionResult` dataclass instance.  Always
            fully populated — never contains ``None`` fields.
            Downstream consumers do not need to check for empty cases.

        Raises:
            RuntimeError: If either resolution stage raises an
                unrecoverable exception.  Individual LLM call failures
                within the detector are handled internally and will not
                propagate here.
        """
        logger.info(
            "TruthResolver: starting resolution on %d chunk(s).", len(chunks)
        )

        if not chunks:
            logger.warning("resolve called with empty chunk list.")
            return ResolutionResult()

        # ---------------------------------------------------------------- #
        # Stage 1: Conflict detection.
        # ---------------------------------------------------------------- #
        try:
            conflicts = self._detector.detect(chunks)
        except Exception as exc:
            raise RuntimeError(
                f"ConflictDetector raised an unrecoverable error: {exc}"
            ) from exc

        # ---------------------------------------------------------------- #
        # Stage 2: Prioritization and suppression.
        # ---------------------------------------------------------------- #
        try:
            prioritization = self._prioritizer.resolve(
                chunks=chunks,
                conflicts=conflicts,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SourcePrioritizer raised an unrecoverable error: {exc}"
            ) from exc

        result = ResolutionResult(
            final_chunks=prioritization["final_chunks"],
            conflicts=conflicts,
            suppressed_chunks=prioritization["suppressed_chunks"],
            resolution_log=prioritization["resolution_log"],
        )

        logger.info(
            "TruthResolver complete: %d final chunk(s), "
            "%d conflict(s), %d suppressed.",
            len(result.final_chunks),   # ← was incorrectly result.suppressed_count
            result.conflict_count,
            result.suppressed_count,
        )

        return result