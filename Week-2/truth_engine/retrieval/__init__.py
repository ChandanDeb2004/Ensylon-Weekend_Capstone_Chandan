"""
retrieval/
~~~~~~~~~~

Station 3 of the RAG pipeline: candidate retrieval and re-ranking.

Two-stage retrieval that maximises both recall (finding the right chunks)
and precision (surfacing the most relevant ones):

**Stage 1 — Hybrid Retrieval** (:class:`HybridRetriever`):
    Queries the vector store (semantic) and BM25 store (keyword)
    simultaneously, then merges the two ranked lists into one via
    Reciprocal Rank Fusion.  Returns the top-20 candidates.

**Stage 2 — Re-ranking** (:class:`Reranker`):
    Scores each ``(query, candidate)`` pair with a Cross-Encoder model,
    which reads the query and chunk jointly for far greater accuracy than
    the bi-encoder used in vector search.  Returns the top-5 chunks that
    the LLM will actually see.

Typical usage::

    from retrieval import HybridRetriever, Reranker

    retriever = HybridRetriever(vector_store, bm25_store)
    reranker  = Reranker()

    candidates = retriever.retrieve(query, top_k=20, source_tiers=[1, 2])
    top_chunks = reranker.rerank(query, candidates, top_k=5)
"""

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

__all__ = [
    "HybridRetriever",
    "Reranker",
]