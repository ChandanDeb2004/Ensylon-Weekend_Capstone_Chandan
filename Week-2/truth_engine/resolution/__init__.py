"""
resolution/
~~~~~~~~~~~

Station 4 of the RAG pipeline: the Truth Engine.

The most architecturally novel station — detects contradictions between
sources and resolves them deterministically before any chunk reaches the
LLM context window.

Components
----------
:class:`ConflictDetector`
    Two-stage detector: cosine similarity gates candidate pairs cheaply,
    then a binary LLM prompt classifies only high-similarity cross-source
    pairs.  Keeps LLM API costs proportional to actual conflict density.

:class:`SourcePrioritizer`
    Deterministic tier-based resolver.  No LLM involved — lower tier
    number always wins.  Suppressed chunks are annotated and preserved
    for citation, never deleted.

:class:`TruthResolver`
    Orchestrator.  Single call returns a :class:`ResolutionResult`
    dataclass consumed by all downstream stations.

:class:`ResolutionResult`
    Stable output contract.  Contains final chunks, all detected
    conflicts, suppressed chunks, and a human-readable resolution log.

Typical usage::

    from resolution import TruthResolver, ResolutionResult
    from resolution import ConflictDetector, SourcePrioritizer

    detector    = ConflictDetector(vector_store, llm_callable)
    prioritizer = SourcePrioritizer()
    resolver    = TruthResolver(detector, prioritizer)

    result: ResolutionResult = resolver.resolve(chunks)

    print(result.format_log())
    final_chunks = result.final_chunks
"""

from resolution.conflict_detector import ConflictDetector
from resolution.source_prioritizer import SourcePrioritizer
from resolution.truth_resolver import ResolutionResult, TruthResolver

__all__ = [
    "ConflictDetector",
    "SourcePrioritizer",
    "TruthResolver",
    "ResolutionResult",
]