"""
storage/
~~~~~~~~

Station 2 of the RAG pipeline: dual-index chunk storage.

Provides two complementary search indexes that must be written to
simultaneously and queried together for best retrieval coverage:

- :class:`VectorStore` — ChromaDB-backed semantic search via
  384-dimensional SentenceTransformer embeddings.  Finds chunks that
  are *conceptually similar* to a query even when exact words differ.

- :class:`BM25Store` — rank_bm25-backed keyword search.  Finds chunks
  containing the *exact terms* in a query, critical for technical
  identifiers, error codes, and precise terminology.

Typical usage::

    from storage import BM25Store, VectorStore

    vector_store = VectorStore(persist_directory="./chroma_db")
    bm25_store   = BM25Store(store_path="./bm25_store")

    # Write to both indexes simultaneously.
    vector_store.add_chunks(chunks, metadatas)
    bm25_store.add_chunks(chunks, metadatas)

    # Query each index — results are merged in the retrieval layer.
    semantic_results = vector_store.query("how do I restart the service?")
    keyword_results  = bm25_store.query("ERR_CODE_0x4F2")
"""

from storage.bm25_store import BM25Store
from storage.vector_store import VectorStore

__all__ = [
    "VectorStore",
    "BM25Store",
]