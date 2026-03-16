"""
evaluation/
~~~~~~~~~~~

Station 6 of the RAG pipeline: answer confidence measurement.

Single component with a single responsibility — compute a confidence
score for every candidate answer and enforce the abstention gate before
the LLM is ever called.

:class:`ConfidenceScorer`
    Computes a weighted three-signal confidence score in [0.0, 1.0]
    and determines whether the system should abstain from answering.
    The ``should_abstain`` method is the hallucination firewall — the
    system returns "I don't know" rather than fabricating an answer
    from weak evidence.

Supporting types:

:class:`ConfidenceResult`
    Dataclass holding the final score, signal breakdown, abstain flag,
    and a human-readable explanation.  The stable output contract
    consumed by ``main.py`` and the prompt builder.

:class:`SignalBreakdown`
    Dataclass holding the three individual signal values (rerank, tier,
    agreement) before weighting.  Included in :class:`ConfidenceResult`
    for transparency and debugging.

Signal weights and the abstention threshold are configurable at
construction time.  The defaults reflect the relative importance of
each signal:

    - Rerank quality    50% — strongest predictor of answer relevance.
    - Source tier       30% — structural authority of the source.
    - Source agreement  20% — corroboration across independent sources.

Typical usage::

    from evaluation import ConfidenceScorer, ConfidenceResult

    scorer = ConfidenceScorer(abstention_threshold=0.40)
    result: ConfidenceResult = scorer.should_abstain(final_chunks)

    if result.abstain:
        return f"I don't know based on the available sources.\\n{result.explanation}"

    # Confidence gate passed — safe to call the LLM.
    answer = llm_client(prompt=built.user_prompt, system_prompt=built.system_prompt)
"""

from evaluation.confidence_scorer import (
    ConfidenceResult,
    ConfidenceScorer,
    SignalBreakdown,
)

__all__ = [
    "ConfidenceScorer",
    "ConfidenceResult",
    "SignalBreakdown",
]