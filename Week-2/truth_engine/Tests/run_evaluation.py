import sys
import json
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must come before any project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import RAGSystem
from ingestion import ingest_markdown, ingest_json

# ---------------------------------------------------------------------------
# Silence ChromaDB telemetry version mismatch warning
# ---------------------------------------------------------------------------
import chromadb.telemetry.product.posthog as _chroma_telemetry
import unittest.mock as _mock
_chroma_telemetry.Posthog = _mock.MagicMock()

# ---------------------------------------------------------------------------
# Step 1 — Inspect your JSONL columns so we can build the right template
# ---------------------------------------------------------------------------
support_log_path = ROOT / "support_logs" / "support_logs.jsonl"

if support_log_path.exists():
    # Read just the first row to see what columns actually exist
    sample = pd.read_json(support_log_path, lines=True, nrows=1)
    print("JSONL columns found:", list(sample.columns))
    print("Sample row:")
    print(sample.iloc[0].to_dict())
    print()
else:
    print(f"[ERROR] File not found: {support_log_path}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2 — Build the template from the actual column names printed above
#
# EDIT THIS to match the columns printed in Step 1.
# Example: if your columns are "error", "resolution", "status" use:
#   SUPPORT_TEMPLATE = "Error: {error}. Resolution: {resolution}. Status: {status}."
# ---------------------------------------------------------------------------
SUPPORT_TEMPLATE = "Issue: {issue}. Resolution: {resolution}. Status: {status}."
# ^ CHANGE THIS after running once and seeing your actual column names above

# ---------------------------------------------------------------------------
# Step 3 — Initialise system
# ---------------------------------------------------------------------------
system = RAGSystem(
    reranker_top_k=5,
    hybrid_candidates=20,
    abstention_threshold=0.40,
)

all_chunks: list[str] = []
all_metadatas: list[dict] = []


def collect(
    chunks: list[str],
    source_name: str,
    tier: int,
    section: str = "",
) -> None:
    """Append chunks and aligned metadata to the master lists."""
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadatas.append({
            "source_tier": tier,
            "source_name": source_name,
            "page_number": i + 1 if tier == 1 else 0,
            "section":     section,
        })
    print(f"  Collected {len(chunks)} chunk(s) from '{source_name}' (tier {tier})")


# ---------------------------------------------------------------------------
# Tier 1 — Manuals (these are .md files, NOT pdfs — use ingest_markdown)
# ---------------------------------------------------------------------------
print("── Tier 1: Manuals ──")
for md_file in [
    ROOT / "manuals" / "auth_service.md",
    ROOT / "manuals" / "payment_api.md",
]:
    if md_file.exists():
        collect(ingest_markdown(md_file), md_file.name, tier=1)
    else:
        print(f"  [WARN] Not found, skipping: {md_file}")

# ---------------------------------------------------------------------------
# Tier 2 — Support logs (JSONL with custom template)
# ---------------------------------------------------------------------------
print("\n── Tier 2: Support Logs ──")
if support_log_path.exists():
    chunks = ingest_json(support_log_path, template=SUPPORT_TEMPLATE)
    if chunks:
        collect(chunks, support_log_path.name, tier=2)
    else:
        print("  [WARN] 0 rows rendered — SUPPORT_TEMPLATE column names")
        print("         do not match the JSONL. Update SUPPORT_TEMPLATE above.")

# ---------------------------------------------------------------------------
# Tier 3 — Wiki
# ---------------------------------------------------------------------------
print("\n── Tier 3: Wiki ──")
wiki_file = ROOT / "wiki" / "legacy_wiki.md"
if wiki_file.exists():
    collect(ingest_markdown(wiki_file), wiki_file.name, tier=3)
else:
    print(f"  [WARN] Not found, skipping: {wiki_file}")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
print()
if not all_chunks:
    print("[ERROR] No chunks ingested. Fix file paths and template above.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Write to both indexes
# ---------------------------------------------------------------------------
system._vector_store.reset()   # clear stale single-chunk index from last run
system._bm25_store.reset()

system._vector_store.add_chunks(all_chunks, all_metadatas)
system._bm25_store.add_chunks(all_chunks, all_metadatas)

print(f"Total chunks ingested: {len(all_chunks)}")
print(f"Vector store:          {system._vector_store.count()}")
print(f"BM25 store:            {system._bm25_store.count()}")

# ---------------------------------------------------------------------------
# Load and run queries
# ---------------------------------------------------------------------------
queries_path = ROOT / "tests" / "test_queries.json"
if not queries_path.exists():
    print(f"[ERROR] Not found: {queries_path}")
    sys.exit(1)

with open(queries_path, encoding="utf-8") as f:
    queries = json.load(f)

results = []
for q in queries:
    query_text = q["query"]
    print(f"\nRunning: {query_text}")
    response = system.query(query_text)
    results.append({
        "query":            query_text,
        "answer":           response.answer,
        "abstained":        response.abstained,
        "confidence_score": response.confidence.score,
        "rerank_signal":    response.confidence.signals.rerank_signal,
        "tier_signal":      response.confidence.signals.tier_signal,
        "agreement_signal": response.confidence.signals.agreement_signal,
        "citations":        response.citations,
        "conflict_log":     response.conflict_log,
        "elapsed_seconds":  response.elapsed_seconds,
    })

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
for r in results:
    print("\n" + "=" * 60)
    print(f"QUERY:      {r['query']}")
    print(f"ABSTAINED:  {r['abstained']}")
    print(f"CONFIDENCE: {r['confidence_score']:.3f}  "
          f"(rerank={r['rerank_signal']:.3f}, "
          f"tier={r['tier_signal']:.3f}, "
          f"agreement={r['agreement_signal']:.3f})")
    print(f"ELAPSED:    {r['elapsed_seconds']:.2f}s")
    if r["citations"]:
        print("CITATIONS:")
        for c in r["citations"]:
            print(f"  {c}")
    if r["conflict_log"] and r["conflict_log"] != (
        "No conflicts detected. All sources are consistent."
    ):
        print("CONFLICT LOG:")
        print(f"  {r['conflict_log']}")
    print("ANSWER:")
    print(r["answer"])

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
output_path = ROOT / "tests" / "test_results.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {output_path}")
