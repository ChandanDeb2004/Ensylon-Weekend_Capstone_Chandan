# Failure & Mitigation Report

## 1. Where Does the System Fail?

### 1.1 Scanned or Image-based PDF
In cases where the PDF contains images, diagrams, or charts, the system fails to extract any content from it. This happens because `pdf_ingestor.py` uses the `pdfplumber` library, which is only capable of extracting searchable tables and textual data. 

### 1.2 The Template Mismatch Problem in JSON Ingestion
The JSON ingestion pipeline uses a strict template which contains `{problem_description}` and `{fix}`. In case of a violation to this template, the whole row is skipped with a warning. 

### 1.3 Sparse Retrieval on Short or Ambiguous Queries
Both the vector store and BM25 index perform poorly on very short queries (one or two words) or queries that use terminology not present in the source documents. The hybrid retriever returns candidates that are loosely related at best. The cross-encoder then scores all of them low, the confidence scorer sees a weak rerank signal, and the system correctly abstains. The user gets an "I don't know" response even though the answer technically exists in the documents.

> **The Mitigation Built:** The two-stage retrieval (hybrid + cross-encoder reranking) and the three-signal confidence scorer handle this case correctly by abstaining rather than hallucinating. The system fails safe.

---

## 2. How Was the LLM Prevented From Following the Legacy Wiki When It Was Wrong?

The system uses multiple independent layers to prevent wrong answers from reaching the users:

### Layer 1 — Source Tier Assignment at Ingestion
Every chunk is tagged with a `source_tier` integer at the moment it is ingested. 
* **Tier 1:** Official manuals
* **Tier 2:** Incident logs
* **Tier 3:** Legacy wiki

This tier is stored in ChromaDB and in the BM25 corpus as metadata. It travels with the chunk through every station of the pipeline.

### Layer 2 — Retrieval Filtering
The hybrid retriever accepts a `source_tiers` filter parameter. By default, the system is configured with `SOURCE_TIERS_FILTER = [1, 2]`, which means tier-3 wiki chunks are excluded from ChromaDB queries and from BM25 results entirely before any ranking takes place. 

For cases where a wiki chunk does enter the candidate pool—for example, when `--all-tiers` is passed on the CLI or when tier filtering is relaxed—the conflict detector compares it against tier-1 and tier-2 chunks. In cases where the data is available in the wiki only, it will warn the user that the information derived is from a deprecated source.

### Layer 3 — Confidence Penalty
Even if a wiki chunk survives conflict resolution (no tier-1 counterpart exists to contradict it), `source_prioritizer.py` applies a **20% multiplicative confidence penalty** to all tier-3 chunks. The confidence scorer then sees a degraded tier signal, which reduces the overall score and makes the system more likely to abstain or hedge on wiki-sourced answers.

---

## 3. How Would You Scale This to 10,000 Documents?

Scaling from dozens of documents to 10,000 requires changes at ingestion, storage, retrieval, and infrastructure. The current architecture handles each concern, but some components need upgrading:

### 3.1 Ingestion 
The current pipeline processes documents sequentially. At 10,000 documents, the sequential ingestion could take hours. 
* **The Fix:** Break the process into multiple processes and assign them to different workers running on different threads on the CPU. 

### 3.2 Storage
ChromaDB's `PersistentClient` is a local, single-process store. At 10,000 documents with an average of 20 chunks per document, that is 200,000 vectors. ChromaDB handles this comfortably on a single machine.
* **Next Step for 1,000,000+ Vectors:** The correct move is to migrate to a dedicated vector database server.

### 3.3 BM25 Scaling
The current BM25 implementation (`vector_store.py`) loads the entire corpus into memory as a Python list and rebuilds the index on every update. At 200,000 chunks, this becomes slow (several seconds per rebuild) and memory-intensive.
* **The Fix:** The correct replacement at scale is **Elasticsearch** or **OpenSearch**, which provides distributed BM25 scoring with inverted indexes, horizontal sharding, and real-time updates without full rebuilds.

### 3.4 Embedding — Batch GPU Processing
The current embedding model runs on CPU using `sentence-transformers`. At 200,000 chunks, embedding the entire corpus takes roughly 30-60 minutes on CPU.
* **The Fix:** Run embedding on GPU. `SentenceTransformer.encode()` already accepts a `device` parameter. Passing `device="cuda"` and increasing `batch_size` to `512` reduces embedding time by 10-20x.

### 3.5 Conflict Detection — O(n²) Problem
The current conflict detector computes cosine similarity for every pair of cross-source chunks passed to it. For 5 chunks, that is 10 pairs. For 200 chunks, that is 19,900 pairs. The cosine comparisons are fast (NumPy), but the LLM calls for confirmed candidates grow linearly with the number of high-similarity pairs found.

### 3.6 Lack of an Interactive Front End
Currently, this system works as a command-line tool and is quite cumbersome to operate. 
* **The Fix:** Building an interactive frontend along with proper error handling will improve the appeal of such a system, making it more user-friendly.
