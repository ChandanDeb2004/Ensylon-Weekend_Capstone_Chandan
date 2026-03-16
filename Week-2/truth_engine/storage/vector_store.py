# ChromaDB init, upsert, query
"""
storage/vector_store.py

Manages the ChromaDB vector index for semantic (meaning-based) search.

Each chunk is embedded into a 384-dimensional vector using a
SentenceTransformer model and stored alongside its metadata.
Retrieval supports optional source_tier filtering so that lower-priority
sources can be deprioritised before results reach the Truth Resolver.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME: str = "rag_chunks"
DEFAULT_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A single query result: the chunk text, its metadata, and its distance.
QueryResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(text: str) -> str:
    """Produce a stable, deterministic ID for a chunk via MD5.

    Using content-derived IDs makes ``upsert`` idempotent — re-ingesting
    the same chunk will overwrite rather than duplicate.

    Args:
        text: The raw chunk string.

    Returns:
        A 32-character lowercase hexadecimal MD5 digest.
    """
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata values to ChromaDB-safe scalar types.

    ChromaDB accepts only ``str``, ``int``, ``float``, and ``bool``.
    ``None`` values and any other types are coerced to their string
    representation to prevent silent storage failures.

    Args:
        metadata: Raw metadata dict, potentially containing ``None``
            or non-scalar values.

    Returns:
        A new dict with all values safe for ChromaDB storage.
    """
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif value is None:
            safe[key] = ""
        else:
            safe[key] = str(value)
    return safe


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class VectorStore:
    """Semantic vector index backed by ChromaDB and SentenceTransformers.

    Chunks are embedded on write and stored with their metadata.
    Retrieval uses cosine similarity with optional metadata filtering.

    Attributes:
        model_name: Name of the SentenceTransformer model in use.
        collection_name: Name of the ChromaDB collection.

    Example:
        >>> store = VectorStore(persist_directory="./chroma_db")
        >>> store.add_chunks(
        ...     chunks=["Restart the daemon to apply changes."],
        ...     metadatas=[{
        ...         "source_tier": 1,
        ...         "source_name": "ops_manual.pdf",
        ...         "page_number": 4,
        ...         "section": "Restart Procedures",
        ...     }],
        ... )
        >>> results = store.query("how do I restart the service?", top_k=3)
    """

    def __init__(
        self,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        in_memory: bool = False,
    ) -> None:
        """Initialise the vector store, loading or creating the collection.

        Args:
            persist_directory: Path where ChromaDB writes its on-disk
                index.  Ignored when ``in_memory=True``.
            collection_name: Name of the ChromaDB collection to use or
                create.
            model_name: HuggingFace model identifier passed to
                :class:`~sentence_transformers.SentenceTransformer`.
            batch_size: Number of chunks to embed per forward pass.
                Larger values are faster but consume more memory.
            in_memory: If ``True``, use an ephemeral in-memory ChromaDB
                client.  Intended for unit tests.
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self._batch_size = batch_size

        logger.info("Loading embedding model '%s' …", model_name)
        self._model: SentenceTransformer = SentenceTransformer(model_name)

        if in_memory:
            logger.info("Using ephemeral in-memory ChromaDB client.")
            self._client = chromadb.Client()
        else:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            logger.info("ChromaDB persisting to '%s'.", persist_path)
            self._client = chromadb.PersistentClient(path=str(persist_path))

        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity metric
        )
        logger.info(
            "Collection '%s' ready. Current document count: %d.",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Embed and upsert a list of text chunks into the vector store.

        Uses ``upsert`` so that re-running the pipeline on unchanged
        documents is idempotent — existing chunks are overwritten rather
        than duplicated.

        Chunks are embedded in batches of ``batch_size`` for efficiency.

        Args:
            chunks: List of plain-text chunk strings to store.
            metadatas: Optional list of metadata dicts, one per chunk.
                Each dict may contain keys: ``source_tier`` (int),
                ``source_name`` (str), ``page_number`` (int),
                ``section`` (str).  Must be the same length as
                ``chunks`` if provided.

        Raises:
            ValueError: If ``metadatas`` is provided but its length
                differs from ``chunks``.
        """
        if not chunks:
            logger.warning("add_chunks called with an empty chunk list. No-op.")
            return

        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks but "
                f"{len(metadatas)} metadata dicts."
            )

        # Fall back to empty metadata dicts if none provided.
        resolved_metadatas: list[dict[str, Any]] = metadatas or [
            {} for _ in chunks
        ]

        ids: list[str] = [_chunk_id(chunk) for chunk in chunks]
        safe_metadatas: list[dict[str, Any]] = [
            _sanitize_metadata(m) for m in resolved_metadatas
        ]

        logger.info(
            "Embedding %d chunk(s) in batches of %d …",
            len(chunks),
            self._batch_size,
        )

        # Embed in batches; convert to nested Python lists for ChromaDB.
        embeddings = self._model.encode(
            chunks,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=safe_metadatas,
        )

        logger.info(
            "Upserted %d chunk(s) into collection '%s'. "
            "Total documents: %d.",
            len(chunks),
            self.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_tiers: list[int] | None = None,
    ) -> list[QueryResult]:
        """Retrieve the top-k semantically similar chunks.

        Args:
            query_text: The natural-language query string.
            top_k: Maximum number of results to return.
            source_tiers: Optional whitelist of ``source_tier`` integer
                values.  When provided, only chunks whose metadata
                ``source_tier`` is in this list are considered.
                Example: ``[1, 2]`` excludes tier-3 (Markdown wiki)
                chunks from results.

        Returns:
            A list of dicts, each containing:

            - ``"text"`` (str): The chunk content.
            - ``"metadata"`` (dict): The stored metadata.
            - ``"distance"`` (float): Cosine distance (lower = closer).
            - ``"id"`` (str): The chunk's MD5 content ID.

            Results are ordered by ascending distance (most similar
            first).  Returns an empty list if the collection is empty
            or no results pass the filter.

        Raises:
            ValueError: If ``query_text`` is empty.
        """
        if not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")

        where_filter: dict | None = None
        if source_tiers:
            where_filter = {"source_tier": {"$in": source_tiers}}

        query_embedding = self._model.encode(
            [query_text],
            convert_to_numpy=True,
        ).tolist()

        raw = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        results: list[QueryResult] = []
        # ChromaDB returns nested lists (one list per query).
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        for doc, meta, dist, chunk_id in zip(
            documents, metadatas, distances, ids
        ):
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "id": chunk_id,
                }
            )

        logger.debug(
            "Query returned %d result(s) for: '%s'",
            len(results),
            query_text[:80],
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of documents in the collection.

        Returns:
            Integer count of stored chunks.
        """
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection, wiping all stored data.

        Intended for testing and pipeline re-runs from scratch.
        """
        logger.warning(
            "Resetting collection '%s'. All data will be lost.",
            self.collection_name,
        )
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def get_embeddings_by_texts(
        self, texts: list[str]
    ) -> dict[str, list[float]]:
        """Retrieve stored embeddings keyed by chunk text.

        Args:
            texts: List of chunk text strings to look up.

        Returns:
            Dict mapping chunk text → embedding vector.
            Texts not found in the store are silently omitted.
        """
        ids = [_chunk_id(t) for t in texts]
        id_to_text = {_chunk_id(t): t for t in texts}
        raw = self._collection.get(ids=ids, include=["embeddings"])
        return {
            id_to_text[doc_id]: emb
            for doc_id, emb in zip(raw["ids"], raw["embeddings"])
            if doc_id in id_to_text
        }