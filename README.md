# Truth Engine
### A Production-Grade RAG Pipeline with Conflict Resolution

> *Retrieves. Resolves. Answers. Honestly.*

Truth Engine is a multi-source Retrieval-Augmented Generation (RAG) system that ingests documents from multiple sources, detects and resolves contradictions between them using a source-tier hierarchy, and generates grounded, cited answers — refusing to answer when evidence is insufficient.

---

## The Problem It Solves

Most RAG systems treat all sources equally. If your official manual says a procedure takes **5 minutes** and your legacy wiki says it takes **15 minutes**, a naive system will either hallucinate a reconciliation or pick one value arbitrarily.

Truth Engine solves this by:

1. **Ranking sources by authority** — official manuals beat support logs beat wikis
2. **Detecting contradictions** before they reach the LLM
3. **Suppressing lower-authority answers** and logging why
4. **Refusing to answer** when retrieved evidence is too weak

---

## Architecture — The 7 Stations

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRUTH ENGINE                             │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  STATION │    │  STATION │    │  STATION │    │  STATION │  │
│  │    1     │───▶│    2     │───▶│   3A+3B  │───▶│    4     │  │
│  │Ingestion │    │ Storage  │    │Retrieval │    │Resolution│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                        │        │
│                  ┌──────────┐    ┌──────────┐         │        │
│                  │  FINAL   │    │  STATION │    ┌────▼─────┐  │
│                  │  ANSWER  │◀───│    7     │◀───│ STATION  │  │
│                  │          │    │   LLM    │    │   5+6    │  │
│                  └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

| Station | Module | Job |
|---------|--------|-----|
| 1 | `ingestion/` | Read PDFs, JSON/CSV, Markdown → clean text chunks |
| 2 | `storage/` | Store chunks in vector index + BM25 keyword index |
| 3A | `retrieval/hybrid_retriever.py` | Query both indexes, merge with RRF |
| 3B | `retrieval/reranker.py` | Cross-encoder reranks top 20 → top 5 |
| 4 | `resolution/` | Detect conflicts, suppress lower-tier sources |
| 5 | `evaluation/` | Score confidence 0–1, abstain if too low |
| 6 | `generation/prompt_builder.py` | Build structured 4-section prompt |
| 7 | `generation/llm_client.py` | Send to Gemini → Groq → Anthropic fallback |

---

## What Makes It Different

### Dual-Index Hybrid Search
Two completely different search methods run simultaneously and their results are merged using **Reciprocal Rank Fusion (RRF)**:

- **Vector search** (ChromaDB + SentenceTransformers) — finds chunks *semantically similar* to the query, even if the exact words differ
- **BM25 keyword search** — finds chunks containing the *exact terms*, critical for error codes like `ERR_CODE_0x4F2`

Neither alone is sufficient. Together they cover each other's blind spots.

### The Truth Engine Core — Conflict Resolution
This is what separates Truth Engine from a standard RAG pipeline.

```
Chunk A (manual.pdf, Tier 1):  "Daemon restart takes 5 minutes."
Chunk B (wiki.md,   Tier 3):   "Daemon restart takes 15 minutes."

     Stage 1: Cosine similarity = 0.91  ──▶  Above 0.85 threshold
     Stage 2: LLM binary check  ──▶  CONFLICT detected

     Resolution: Tier 1 WINS
     wiki.md chunk: suppressed, logged, preserved for citation

     Final answer cites manual.pdf and discloses the override.
```

The conflict detection uses two stages to keep LLM API costs proportional to actual conflict density — cheap cosine comparisons gate which pairs ever reach the LLM.

### The Hallucination Firewall
Before any prompt is built or LLM called, the system scores confidence from three signals:

| Signal | Weight | Measures |
|--------|--------|---------|
| Rerank quality | 50% | How relevant is the best retrieved chunk? |
| Source tier | 30% | Was this from an authoritative source? |
| Source agreement | 20% | Did multiple independent sources agree? |

If the combined score is below **0.40**, the system abstains entirely and returns *"I don't know based on the available sources"* rather than fabricating an answer.

---

## Project Structure

```
truth_engine/
│
├── main.py                          # Director — orchestrates all stations
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_ingestor.py              # PDF + table extraction (pdfplumber)
│   ├── json_ingestor.py             # JSON/CSV → human-readable sentences
│   ├── markdown_ingestor.py         # Header-aware Markdown splitting
│   └── ingestion_pipeline.py        # Orchestrator + MD5 deduplication
│
├── storage/
│   ├── __init__.py
│   ├── vector_store.py              # ChromaDB + SentenceTransformer embeddings
│   └── bm25_store.py                # rank_bm25 keyword index + JSON persistence
│
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py          # RRF fusion of vector + BM25 results
│   └── reranker.py                  # Cross-encoder reranking (top 20 → top 5)
│
├── resolution/
│   ├── __init__.py
│   ├── conflict_detector.py         # Cosine gate + LLM binary classification
│   ├── source_prioritizer.py        # Deterministic tier-based resolution
│   └── truth_resolver.py            # Resolution orchestrator
│
├── evaluation/
│   ├── __init__.py
│   └── confidence_scorer.py         # 3-signal weighted score + abstention gate
│
├── generation/
│   ├── __init__.py
│   ├── prompt_builder.py            # 4-section structured prompt construction
│   └── llm_client.py                # Gemini → Groq → Anthropic fallback client
│
├── tests/
│   ├── test_queries.json            # Query inputs for evaluation
│   ├── test_results.json            # Generated outputs (gitignored)
│   └── run_evaluation.py            # Evaluation runner script
│
├── requirements.txt
├── .gitignore
└── FAILURE_AND_MITIGATION.md
```

---

## Installation

**Requirements:** Python 3.10 or higher

```bash
# 1. Clone the repository
git clone https://github.com/your-username/truth-engine.git
cd truth-engine

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (never commit this):

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here   # optional third fallback
```

Load it before running:

```bash
# Windows PowerShell
Get-Content .env | ForEach-Object { $name, $value = $_ -split '=', 2; Set-Item "env:$name" $value }

# macOS / Linux
export $(cat .env | xargs)
```

---

## Usage

### Command Line

```bash
# Ingest documents
python main.py --ingest \
  --pdf  "manuals/payment_api.pdf" \
  --json "support_logs/incidents.csv" \
  --markdown "wiki/procedures.md"

# Query the system
python main.py --query "Why does the payment API return HTTP 503?"

# Ingest and query in one command
python main.py \
  --ingest \
  --pdf "manuals/payment_api.pdf" \
  --query "Why does the payment API return HTTP 503?"

# Include tier-3 wiki content (excluded by default)
python main.py --query "What do the legacy procedures say?" --all-tiers

# Lower the abstention threshold (more willing to answer)
python main.py --query "What is the restart time?" --threshold 0.30
```


---

## Example Output

```
══════════════════════════════════════════════════════════════════════
  QUERY: Why does the payment API return HTTP 503?
══════════════════════════════════════════════════════════════════════
  ANSWER:

  The payment API returns HTTP 503 during peak traffic when the upstream
  payment processor connection pool is exhausted [SRC-1]. This is a
  known issue documented in incident INC-0442, where the fix was to
  increase the connection pool size from 10 to 50 [SRC-2].

──────────────────────────────────────────────────────────────────────
  CITATIONS:
    [SRC-1] According to payment_api.pdf (Page 7, §Error Codes)
    [SRC-2] According to incidents.csv
──────────────────────────────────────────────────────────────────────
  CONFLICT RESOLUTION LOG:
  CONFLICT RESOLVED — 'payment_api.pdf' (tier 1) overrides
  'legacy_wiki.md' (tier 3). Reason: outranked_by_tier_1.
  Similarity: 0.9102.
──────────────────────────────────────────────────────────────────────
  CONFIDENCE: 0.891  (rerank=0.985, tier=1.000, agreement=0.889)
  ELAPSED:    2.34s
══════════════════════════════════════════════════════════════════════
```

### Abstention Example

When evidence is insufficient, the system refuses to answer rather than fabricating a response:

```
══════════════════════════════════════════════════════════════════════
  QUERY: How can I reset the database cache?
══════════════════════════════════════════════════════════════════════
  ⚠  SYSTEM ABSTAINED — confidence below threshold.

  I don't know based on the available sources.

  Confidence score: 0.003 (threshold: 0.40).
    Rerank signal:    0.033 (weight 50%)
    Tier signal:      0.800 (weight 30%)
    Agreement signal: 0.000 (weight 20%)
  Decision: ABSTAIN — Weakest signal: rerank_signal (0.033).
══════════════════════════════════════════════════════════════════════
```

---

## Source Tiers

The tier system is the foundation of conflict resolution. Assign tiers at ingestion based on how authoritative each source is.

| Tier | Authority | Example Sources | Confidence Penalty |
|------|-----------|-----------------|-------------------|
| 1 | Official | Product manuals, technical specs, API docs | None |
| 2 | Secondary | Incident logs, support tickets, resolved issues | −10% |
| 3 | Tertiary | Internal wikis, community notes, legacy docs | −20% |

**Lower number always wins.** A tier-1 manual will always override a tier-3 wiki on the same fact, regardless of how confidently the wiki states it.

By default, **tier-3 sources are excluded from retrieval entirely** and only included when `--all-tiers` is passed. They can only influence answers when no tier-1 or tier-2 source covers the same topic.

---

## JSON/CSV Template Configuration

When ingesting structured log data, supply a template matching your column names:

```python
# Check your actual column names first
import pandas as pd
df = pd.read_json("support_logs/incidents.jsonl", lines=True, nrows=1)
print(df.columns.tolist())
# Output: ['ticket_id', 'error_message', 'resolution', 'status']

# Then build a matching template
from ingestion import ingest_json

chunks = ingest_json(
    "support_logs/incidents.jsonl",
    template="Error: {error_message}. Resolution: {resolution}. Status: {status}."
)
```

---

## Running the Evaluation Suite

```bash
# Run all test queries and save results
python tests/run_evaluation.py

# Results are saved to tests/test_results.json
```

Test queries are defined in `tests/test_queries.json`:

```json
[
  { "query": "Why does the payment API return HTTP 503?" },
  { "query": "How do I enable metrics for microservices?" },
  { "query": "What is the restart procedure for the auth service?" }
]
```

---

## LLM Provider Fallback Chain

The system tries providers in order and falls back automatically on failure:

```
Gemini (primary)
    │
    ├── Success ──▶ return answer
    │
    └── All retries failed
            │
            ▼
        Groq (fallback)
            │
            ├── Success ──▶ return answer
            │
            └── All retries failed
                    │
                    ▼
                Anthropic (last resort)
```

Each provider implements **exponential backoff with jitter** — 3 retry attempts, delays of 1s → 2s → 4s with ±0.5s random jitter to prevent thundering herd on rate limits.

---

## Key Design Principles

**Every station has one job.** `ingestion/` knows nothing about storage. `storage/` knows nothing about retrieval. `retrieval/` knows nothing about resolution. Each module's input and output type is documented and stable.

**Dependency injection throughout.** The `ConflictDetector` accepts any `LLMCallable`. The `HybridRetriever` accepts any store objects. Tests can inject mocks without touching production code.

**Fail safe, not fail loud.** The confidence scorer abstains rather than hallucinating. The ingestion pipeline skips bad files rather than crashing. The LLM client returns a graceful error string rather than raising. The system degrades to partial results, never to silent fabrication.

**Suppress, never delete.** Losing chunks in conflict resolution are annotated and preserved. The system can always explain what it overrode and why.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pdfplumber` | ~0.11 | PDF text and table extraction |
| `pandas` | ~2.2 | JSON/CSV ingestion |
| `langchain-text-splitters` | ~0.3 | Header-aware Markdown splitting |
| `chromadb` | ~0.5 | Vector database |
| `sentence-transformers` | ~3.0 | Bi-encoder embeddings + cross-encoder reranking |
| `rank-bm25` | ~0.2 | BM25 keyword index |
| `numpy` | ~1.26 | Cosine similarity computation |
| `google-generativeai` | ~0.8 | Gemini LLM client |
| `groq` | ~0.9 | Groq LLM client |
| `anthropic` | ~0.30 | Anthropic LLM client (fallback) |

---

## Known Limitations

- **Scanned PDFs** produce no extractable text and are skipped. An OCR fallback (`pytesseract`) is the correct next step for legacy document archives.
- **Conflict detection** requires cosine similarity above 0.85. Contradictions expressed in very different language may not be caught.
- **BM25 is case-sensitive** by design to preserve exact technical term matching. Queries must use the correct case for error codes and identifiers.
- **Single-chunk indexes** skip conflict detection. The system needs at least two chunks from different sources to detect anything.

See [`FAILURE_AND_MITIGATION.md`](Week-2/truth_engine/README.md) for a full analysis of failure modes and their mitigations.

---

## Scaling to Production

The current architecture handles thousands of documents on a single machine. For 10,000+ documents:

- Replace `BM25Store` with **Elasticsearch** for distributed keyword search
- Replace `ChromaDB PersistentClient` with **Pinecone or Qdrant** for managed vector search
- Move conflict detection to an **offline batch job** during ingestion (not live at query time)
- Wrap `RAGSystem` in **FastAPI** for concurrent query serving
- Run embedding on **GPU** with `batch_size=512`

The clean interface boundaries mean each swap is contained to one file.

---

*Built as a demonstration of production RAG architecture with source conflict resolution.*
