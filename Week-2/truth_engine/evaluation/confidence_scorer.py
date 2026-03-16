# Computes & returns confidence scores
"""
evaluation/confidence_scorer.py

Computes a single confidence score in [0, 1] for each system answer,
and determines whether the system should abstain from answering.

The score combines three independent signals via a weighted formula:

    Signal 1 — Rerank quality    (weight: 0.50)
        Sigmoid-normalised top cross-encoder score.  Measures how
        relevant the best retrieved chunk is to the query.

    Signal 2 — Source tier       (weight: 0.30)
        Penalty derived from the lowest-authority tier among the chunks
        used.  Mirrors the tier penalties in SourcePrioritizer.

    Signal 3 — Source agreement  (weight: 0.20)
        Tier-weighted count of independent sources that contributed
        chunks.  Multiple corroborating sources increase confidence.

If the final score falls below the abstention threshold, ``should_abstain``
returns True and the system emits an "I don't know" response instead of
calling the LLM — the hallucination firewall.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — import from config.py in production.
# ---------------------------------------------------------------------------

# Weights for the three confidence signals.  Must sum to 1.0.
RERANK_WEIGHT: float = 0.50
TIER_WEIGHT: float = 0.30
AGREEMENT_WEIGHT: float = 0.20

from config import TIER_PENALTIES, DEFAULT_TIER_PENALTY, ABSTENTION_THRESHOLD



# Normalisation constant for tier-weighted agreement.
# Set to the maximum achievable agreement score (two tier-1 sources).
_AGREEMENT_NORM: float = 1.0 + 0.5   # 1/1 + 1/2

# Minimum rerank score below which the rerank signal fast-path abstains.
RERANK_FLOOR: float = -3.0


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

@dataclass
class SignalBreakdown:
    """Individual signal values contributing to the confidence score.

    Attributes:
        rerank_signal: Sigmoid-normalised top cross-encoder score.
        tier_signal: Source tier quality score (1.0 = all tier-1).
        agreement_signal: Tier-weighted source agreement score.
    """

    rerank_signal: float
    tier_signal: float
    agreement_signal: float

    def weakest(self) -> tuple[str, float]:
        """Return the name and value of the weakest signal.

        Returns:
            A 2-tuple of ``(signal_name, value)`` for the signal with
            the lowest contribution to the final score, weighted by its
            configured weight.

        Example:
            >>> breakdown.weakest()
            ('agreement_signal', 0.18)
        """
        weighted = {
            "rerank_signal":    self.rerank_signal    * RERANK_WEIGHT,
            "tier_signal":      self.tier_signal      * TIER_WEIGHT,
            "agreement_signal": self.agreement_signal * AGREEMENT_WEIGHT,
        }
        name = min(weighted, key=weighted.__getitem__)
        return name, getattr(self, name)


@dataclass
class ConfidenceResult:
    """Complete confidence assessment for a single answer.

    Attributes:
        score: Final weighted confidence score in [0.0, 1.0].
        signals: Breakdown of individual signal values.
        abstain: Whether the system should abstain from answering.
        explanation: Human-readable explanation of the score and, if
            abstaining, which signal was weakest.
        threshold: The abstention threshold used for this assessment.
    """

    score: float
    signals: SignalBreakdown
    abstain: bool
    explanation: str
    threshold: float = ABSTENTION_THRESHOLD

    def __str__(self) -> str:
        status = "ABSTAIN" if self.abstain else "ANSWER"
        return (
            f"ConfidenceResult({status} | score={self.score:.3f} | "
            f"rerank={self.signals.rerank_signal:.3f}, "
            f"tier={self.signals.tier_signal:.3f}, "
            f"agreement={self.signals.agreement_signal:.3f})"
        )


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Apply the sigmoid function to map a real value to (0.0, 1.0).

    Used to normalise cross-encoder logit scores, which have no fixed
    range, into a probability-like signal.

    Args:
        x: Input value (cross-encoder logit score).

    Returns:
        Value in the open interval (0.0, 1.0).
    """
    return 1.0 / (1.0 + math.exp(-x))


def _compute_rerank_signal(chunks: list[dict[str, Any]]) -> float:
    """Compute the rerank quality signal from cross-encoder scores.

    Takes the top (highest) cross-encoder score across all chunks and
    applies sigmoid normalisation to map it into [0.0, 1.0].

    Args:
        chunks: List of chunk dicts.  Each may contain a
            ``"cross_encoder_score"`` key from the re-ranker.

    Returns:
        Sigmoid-normalised top cross-encoder score, or ``0.0`` if no
        chunks contain a cross-encoder score.
    """
    scores: list[float] = [
        c["cross_encoder_score"]
        for c in chunks
        if "cross_encoder_score" in c
    ]
    if not scores:
        logger.debug("No cross_encoder_score found in chunks; rerank signal = 0.0.")
        return 0.0

    top_score = max(scores)
    normalised = _sigmoid(top_score)
    logger.debug(
        "Rerank signal: top_score=%.4f → sigmoid=%.4f", top_score, normalised
    )
    return normalised


def _compute_tier_signal(chunks: list[dict[str, Any]]) -> float:
    """Compute the source tier quality signal.

    The weakest (highest-numbered) tier among all chunks determines the
    penalty.  A single tier-3 chunk in the context degrades the signal
    for the whole answer.

    Args:
        chunks: List of chunk dicts with ``"metadata"`` containing
            ``"source_tier"`` values.

    Returns:
        Tier quality score in [0.0, 1.0].  1.0 means all chunks are
        tier 1 (no penalty).  Lower values reflect lower-authority
        sources.
    """
    tiers: list[int] = [
        c.get("metadata", {}).get("source_tier", 99)
        for c in chunks
    ]
    if not tiers:
        return 1.0 - DEFAULT_TIER_PENALTY

    worst_tier = max(tiers)
    penalty = TIER_PENALTIES.get(worst_tier, DEFAULT_TIER_PENALTY)
    signal = 1.0 - penalty
    logger.debug(
        "Tier signal: worst_tier=%d, penalty=%.2f → signal=%.4f",
        worst_tier, penalty, signal,
    )
    return signal


def _compute_agreement_signal(chunks: list[dict[str, Any]]) -> float:
    """Compute the tier-weighted source agreement signal.

    Counts unique source documents and weights each by its tier
    authority (lower tier = higher weight).  Multiple independent
    authoritative sources corroborating the same answer increases
    confidence; a single source cannot produce a maximum agreement score.

    Formula::

        agreement = min(
            sum(1 / tier for each unique source) / AGREEMENT_NORM,
            1.0
        )

    Args:
        chunks: List of chunk dicts with ``"metadata"`` containing
            ``"source_name"`` and ``"source_tier"`` keys.

    Returns:
        Agreement signal in [0.0, 1.0].
    """
    # Map source_name → lowest tier seen for that source.
    source_tiers: dict[str, int] = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        name: str = meta.get("source_name", "")
        tier: int = meta.get("source_tier", 99)
        if name:
            # Keep the best (lowest) tier if the same source appears
            # multiple times with different tier values (shouldn't happen
            # in practice, but defensive coding here is cheap).
            source_tiers[name] = min(source_tiers.get(name, 99), tier)

    if not source_tiers:
        return 0.0

    weighted_sum: float = sum(
        1.0 / max(tier, 1)   # guard against tier=0 division
        for tier in source_tiers.values()
    )
    signal = min(weighted_sum / _AGREEMENT_NORM, 1.0)
    logger.debug(
        "Agreement signal: sources=%s → weighted_sum=%.4f → signal=%.4f",
        dict(source_tiers),
        weighted_sum,
        signal,
    )
    return signal


def _build_explanation(
    score: float,
    signals: SignalBreakdown,
    abstain: bool,
    threshold: float,
) -> str:
    """Build a human-readable explanation of the confidence assessment.

    Args:
        score: Final weighted confidence score.
        signals: Individual signal breakdown.
        abstain: Whether the system is abstaining.
        threshold: The abstention threshold.

    Returns:
        A concise explanation string suitable for inclusion in the
        final user-facing response or system log.
    """
    lines: list[str] = [
        f"Confidence score: {score:.3f} "
        f"(threshold: {threshold:.2f}).",
        f"  Rerank signal:    {signals.rerank_signal:.3f} "
        f"(weight {RERANK_WEIGHT:.0%})",
        f"  Tier signal:      {signals.tier_signal:.3f} "
        f"(weight {TIER_WEIGHT:.0%})",
        f"  Agreement signal: {signals.agreement_signal:.3f} "
        f"(weight {AGREEMENT_WEIGHT:.0%})",
    ]

    if abstain:
        weak_name, weak_value = signals.weakest()
        lines.append(
            f"Decision: ABSTAIN — score below threshold. "
            f"Weakest signal: {weak_name} ({weak_value:.3f})."
        )
    else:
        lines.append("Decision: ANSWER — score meets threshold.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Computes weighted confidence scores and enforces the abstention gate.

    The abstention gate is the hallucination firewall: when the computed
    confidence score falls below ``abstention_threshold``, the system
    declines to call the LLM rather than risk generating a confidently-
    worded answer from weak or irrelevant evidence.

    Attributes:
        abstention_threshold: Score below which ``should_abstain`` fires.
        weights: The (rerank, tier, agreement) weight tuple in use.

    Example:
        >>> scorer = ConfidenceScorer()
        >>> result = scorer.score(chunks)
        >>> if result.abstain:
        ...     print(f"Abstaining: {result.explanation}")
        ... else:
        ...     answer = llm_client(prompt)
    """

    def __init__(
        self,
        abstention_threshold: float = ABSTENTION_THRESHOLD,
        rerank_weight: float = RERANK_WEIGHT,
        tier_weight: float = TIER_WEIGHT,
        agreement_weight: float = AGREEMENT_WEIGHT,
    ) -> None:
        """Initialise the scorer with configurable weights and threshold.

        Args:
            abstention_threshold: Confidence score below which the system
                should abstain.  Must be in [0.0, 1.0].
            rerank_weight: Weight for the rerank signal.
            tier_weight: Weight for the tier signal.
            agreement_weight: Weight for the agreement signal.

        Raises:
            ValueError: If weights do not sum to 1.0 (within floating
                point tolerance), or if ``abstention_threshold`` is
                outside [0.0, 1.0].
        """
        weight_sum = rerank_weight + tier_weight + agreement_weight
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got "
                f"{rerank_weight} + {tier_weight} + {agreement_weight} "
                f"= {weight_sum:.6f}."
            )
        if not (0.0 <= abstention_threshold <= 1.0):
            raise ValueError(
                f"abstention_threshold must be in [0.0, 1.0], "
                f"got {abstention_threshold}."
            )

        self.abstention_threshold = abstention_threshold
        self.weights = (rerank_weight, tier_weight, agreement_weight)
        self._rerank_weight = rerank_weight
        self._tier_weight = tier_weight
        self._agreement_weight = agreement_weight

        logger.info(
            "ConfidenceScorer initialised: threshold=%.2f, "
            "weights=(rerank=%.2f, tier=%.2f, agreement=%.2f).",
            abstention_threshold,
            rerank_weight,
            tier_weight,
            agreement_weight,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fast_path_abstain(
        self, chunks: list[dict[str, Any]]
    ) -> ConfidenceResult | None:
        """Check fast-path abstention conditions before full scoring.

        Guards against degenerate inputs that warrant immediate
        abstention without computing all three signals.

        Fast-path cases:
        - Empty chunk list: no evidence at all.
        - Top rerank score below ``RERANK_FLOOR``: the best chunk is
          strongly irrelevant.

        Args:
            chunks: The list of final chunks to assess.

        Returns:
            A :class:`ConfidenceResult` with ``abstain=True`` if a
            fast-path condition is met, otherwise ``None``.
        """
        if not chunks:
            explanation = (
                "Abstaining: no chunks available. "
                "Retrieval returned no results for this query."
            )
            logger.warning(explanation)
            return ConfidenceResult(
                score=0.0,
                signals=SignalBreakdown(0.0, 0.0, 0.0),
                abstain=True,
                explanation=explanation,
                threshold=self.abstention_threshold,
            )

        top_score: float = max(
            (c.get("cross_encoder_score", 0.0) for c in chunks),
            default=0.0,
        )
        if top_score < RERANK_FLOOR:
            explanation = (
                f"Abstaining: top rerank score {top_score:.4f} is below "
                f"the floor {RERANK_FLOOR}. "
                "Retrieved chunks are strongly irrelevant to the query."
            )
            logger.warning(explanation)
            zero_signals = SignalBreakdown(
                rerank_signal=_sigmoid(top_score),
                tier_signal=0.0,
                agreement_signal=0.0,
            )
            return ConfidenceResult(
                score=_sigmoid(top_score) * self._rerank_weight,
                signals=zero_signals,
                abstain=True,
                explanation=explanation,
                threshold=self.abstention_threshold,
            )

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, chunks: list[dict[str, Any]]) -> ConfidenceResult:
        """Compute the full confidence score for a set of answer chunks.

        Runs fast-path abstention checks first, then computes all three
        signals and combines them with the configured weights.

        Args:
            chunks: List of chunk dicts as returned by
                :class:`~resolution.TruthResolver` (or its
                ``final_chunks`` list).  Each dict should contain:

                - ``"cross_encoder_score"`` (float): From the re-ranker.
                - ``"metadata"`` (dict): With ``"source_tier"`` (int)
                  and ``"source_name"`` (str).
                - ``"confidence"`` (float, optional): Penalty already
                  applied by :class:`~resolution.SourcePrioritizer`.

        Returns:
            A :class:`ConfidenceResult` dataclass.  Always fully
            populated — ``abstain`` is ``True`` if the score falls
            below ``abstention_threshold`` or a fast-path condition
            is met.
        """
        # ---------------------------------------------------------------- #
        # Fast-path checks — return early if degenerate input detected.
        # ---------------------------------------------------------------- #
        fast_path = self._fast_path_abstain(chunks)
        if fast_path is not None:
            return fast_path

        # ---------------------------------------------------------------- #
        # Compute the three signals.
        # ---------------------------------------------------------------- #
        rerank_signal = _compute_rerank_signal(chunks)
        tier_signal = _compute_tier_signal(chunks)
        agreement_signal = _compute_agreement_signal(chunks)

        signals = SignalBreakdown(
            rerank_signal=round(rerank_signal, 6),
            tier_signal=round(tier_signal, 6),
            agreement_signal=round(agreement_signal, 6),
        )

        # ---------------------------------------------------------------- #
        # Weighted combination.
        # ---------------------------------------------------------------- #
        score: float = round(
            (rerank_signal    * self._rerank_weight)
            + (tier_signal    * self._tier_weight)
            + (agreement_signal * self._agreement_weight),
            6,
        )
        # Clamp to [0.0, 1.0] to guard against floating point drift.
        score = max(0.0, min(1.0, score))

        abstain = score < self.abstention_threshold
        explanation = _build_explanation(
            score, signals, abstain, self.abstention_threshold
        )

        result = ConfidenceResult(
            score=score,
            signals=signals,
            abstain=abstain,
            explanation=explanation,
            threshold=self.abstention_threshold,
        )

        logger.info(str(result))
        return result

    def should_abstain(self, chunks: list[dict[str, Any]]) -> ConfidenceResult:
        """Evaluate whether the system should abstain from answering.

        Convenience wrapper around :meth:`score` that makes the
        call-site intent explicit.  The return type is identical —
        callers check ``result.abstain`` to branch on the decision.

        This is the hallucination firewall entry point.  Call this
        before building the prompt or invoking the LLM client::

            result = scorer.should_abstain(final_chunks)
            if result.abstain:
                return f"I don't know. {result.explanation}"
            answer = llm_client(prompt)

        Args:
            chunks: List of final chunk dicts from
                :class:`~resolution.TruthResolver`.

        Returns:
            A :class:`ConfidenceResult` with ``abstain`` set
            appropriately.
        """
        return self.score(chunks)