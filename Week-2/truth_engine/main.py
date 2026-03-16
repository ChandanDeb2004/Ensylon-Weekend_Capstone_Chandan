# Entrypoint � CLI & orchestration
"""
main.py

The Director — orchestrates all five stations of the RAG pipeline
in sequence.  Contains no business logic; all decisions are delegated
to the module that owns them.

Pipeline order:
    1. Ingestion      — ingest/ (run once; chunks stored in both indexes)
    2. Retrieval      — retrieval/hybrid_retriever.py
    3. Re-ranking     — retrieval/reranker.py
    4. Resolution     — resolution/truth_resolver.py
    5. Confidence     — evaluation/confidence_scorer.py
                        → abstain if score < threshold
    6. Prompt build   — generation/prompt_builder.py
    7. LLM generation — generation/llm_client.py
    8. Final answer   — structured FinalAnswer dataclass

Usage:
    # Ingest documents, then run a query:
    python main.py --ingest --query "How long does Procedure X take?"

    # Query only (documents already ingested):
    python main.py --query "How long does Procedure X take?"
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Station imports
# ---------------------------------------------------------------------------
from evaluation import ConfidenceResult, ConfidenceScorer
from generation import (
    BuiltPrompt,
    FallbackClient,
    GeminiClient,
    GroqClient,
    PromptBuilder,
)
from ingestion import run_ingestion_pipeline
from resolution import (
    ConflictDetector,
    ResolutionResult,
    SourcePrioritizer,
    TruthResolver,
)
from retrieval import HybridRetriever, Reranker
from storage import BM25Store, VectorStore
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline configuration — centralised defaults.
# Import from config.py in production.
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR: str = "./chroma_db"
BM25_STORE_DIR: str = "./bm25_store"
HYBRID_CANDIDATES: int = 20      # Candidates fetched from each store
RERANKER_TOP_K: int = 5          # Chunks the LLM actually sees
ABSTENTION_THRESHOLD: float = 0.40
SOURCE_TIERS_FILTER: list[int] = [1, 2]   # Tier-3 wiki deprioritised

# JSON/CSV template for incident log records.
INCIDENT_TEMPLATE: str = (
    "Problem: {problem_description}. "
    "Fix Applied: {fix}. "
    "Outcome: {outcome}."
)
# ---------------------------------------------------------------------------
# Helper Function
# ---------------------------------------------------------------------------
def _build_default_metadatas(
    chunks: list[str],
    pdf_paths: list[str | Path],
    json_paths: list[str | Path],
    markdown_paths: list[str | Path],
) -> list[dict]:
    """Build per-chunk metadata by re-running ingestion per source file
    and tracking which chunks came from which file.

    This is used by the CLI path where no explicit metadata is provided.
    Tier assignment:
        pdf_paths      → tier 1
        json_paths     → tier 2
        markdown_paths → tier 3

    Args:
        chunks: The full deduplicated chunk list from the pipeline.
        pdf_paths: PDF source file paths.
        json_paths: JSON/CSV source file paths.
        markdown_paths: Markdown source file paths.

    Returns:
        A list of metadata dicts aligned to ``chunks``.
    """
    from ingestion import ingest_pdf, ingest_json, ingest_markdown

    # Map chunk text → metadata by re-ingesting each file individually.
    # This is safe because ingest_* functions are pure — same file
    # always produces the same chunks.
    text_to_meta: dict[str, dict] = {}

    source_groups = [
        (pdf_paths,      ingest_pdf,      1),
        (json_paths,     ingest_json,     2),
        (markdown_paths, ingest_markdown, 3),
    ]

    for paths, ingest_fn, tier in source_groups:
        for path in paths:
            path = Path(path)
            try:
                file_chunks = ingest_fn(path)
                for i, chunk in enumerate(file_chunks):
                    if chunk not in text_to_meta:
                        text_to_meta[chunk] = {
                            "source_tier": tier,
                            "source_name": path.name,
                            "page_number": i + 1 if tier == 1 else 0,
                            "section":     "",
                        }
            except Exception as exc:
                logger.warning(
                    "Could not re-ingest '%s' for metadata: %s", path.name, exc
                )

    # Align metadata to the deduplicated chunk list.
    # Fall back to a labelled unknown only if a chunk somehow has no match.
    resolved = []
    for chunk in chunks:
        meta = text_to_meta.get(chunk, {
            "source_tier": 1,
            "source_name": "untracked",
            "page_number": 0,
            "section":     "",
        })
        resolved.append(meta)

    return resolved


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class FinalAnswer:
    """Structured output from a single RAG pipeline query.

    This is the stable API contract between ``main.py`` and any consumer
    (CLI display, REST API serialisation, test assertions).

    Attributes:
        query: The original user query string.
        answer: The LLM-generated answer, or an abstention message.
        confidence: The :class:`~evaluation.ConfidenceResult` for this
            answer, including the score and signal breakdown.
        citations: Pre-generated citation strings, one per context chunk
            shown to the LLM.  Empty if the system abstained.
        conflict_log: Human-readable resolution log from
            :class:`~resolution.TruthResolver`.  Empty string if no
            conflicts were detected.
        abstained: ``True`` if the confidence gate fired and the LLM
            was not called.
        elapsed_seconds: Wall-clock time for the full query pipeline
            in seconds.
        source_map: Maps ``"SRC-N"`` identifiers to chunk metadata
            dicts.  Empty if the system abstained.
    """

    query: str
    answer: str
    confidence: ConfidenceResult
    citations: list[str] = field(default_factory=list)
    conflict_log: str = ""
    has_conflicts: bool = False
    abstained: bool = False
    elapsed_seconds: float = 0.0
    source_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def display(self) -> str:
        """Format the answer for human-readable console output.

        Returns:
            A multi-section string containing the answer, citations,
            conflict log (if any), and confidence breakdown.
        """
        divider = "═" * 70
        thin = "─" * 70
        sections: list[str] = [
            divider,
            f"  QUERY: {self.query}",
            divider,
        ]

        if self.abstained:
            sections += [
                "  ⚠  SYSTEM ABSTAINED — confidence below threshold.",
                "",
                self.answer,
            ]
        else:
            sections += ["  ANSWER:", "", self.answer]

        sections.append(thin)

        if self.citations:
            sections.append("  CITATIONS:")
            for citation in self.citations:
                sections.append(f"    {citation}")
            sections.append(thin)

        

        if self.has_conflicts:   # ← clean boolean check, no string coupling
            sections += ["  CONFLICT RESOLUTION LOG:", self.conflict_log, thin]

        sections += [
            f"  CONFIDENCE: {self.confidence.score:.3f}  "
            f"(rerank={self.confidence.signals.rerank_signal:.3f}, "
            f"tier={self.confidence.signals.tier_signal:.3f}, "
            f"agreement={self.confidence.signals.agreement_signal:.3f})",
            f"  ELAPSED:    {self.elapsed_seconds:.2f}s",
            divider,
        ]

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# RAG system — wires and holds all components
# ---------------------------------------------------------------------------

class RAGSystem:
    """The RAG pipeline director.

    Instantiates every station component once at construction time and
    exposes two public methods:

    - :meth:`ingest`: Run the ingestion pipeline over source files and
      populate both storage indexes.
    - :meth:`query`: Run the full retrieval-to-generation pipeline for
      a user query and return a :class:`FinalAnswer`.

    All heavy models (embedding, cross-encoder, LLM client) are loaded
    in ``__init__`` so they are ready on the first ``query()`` call
    without per-request initialisation overhead.

    Example:
        >>> system = RAGSystem()
        >>> system.ingest(
        ...     pdf_paths=["docs/manual.pdf"],
        ...     json_paths=["data/incidents.csv"],
        ...     markdown_paths=["wiki/procedures.md"],
        ... )
        >>> answer = system.query("How long does Procedure X take?")
        >>> print(answer.display())
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_PERSIST_DIR,
        bm25_dir: str = BM25_STORE_DIR,
        abstention_threshold: float = ABSTENTION_THRESHOLD,
        source_tiers_filter: list[int] | None = None,
        hybrid_candidates: int = HYBRID_CANDIDATES,
        reranker_top_k: int = RERANKER_TOP_K,
        in_memory: bool = False,
    ) -> None:
        """Initialise and wire all pipeline components.

        Args:
            chroma_dir: Directory for ChromaDB persistence.
            bm25_dir: Directory for BM25 corpus persistence.
            abstention_threshold: Confidence score below which the
                system abstains rather than calling the LLM.
            source_tiers_filter: Whitelist of source tier integers
                passed to the retriever.  ``None`` disables filtering.
            hybrid_candidates: Number of candidates fetched from each
                store before RRF fusion.
            reranker_top_k: Number of chunks the cross-encoder returns
                to the truth resolver and prompt builder.
            in_memory: If ``True``, use an ephemeral in-memory
                ChromaDB client.  Intended for testing.
        """
        self._source_tiers_filter = source_tiers_filter or SOURCE_TIERS_FILTER
        self._hybrid_candidates = hybrid_candidates
        self._reranker_top_k = reranker_top_k

        logger.info("Initialising RAG pipeline components …")
        t_start = time.perf_counter()

        # ---------------------------------------------------------------- #
        # Station 2 — Storage
        # ---------------------------------------------------------------- #
        logger.info("  [2/7] Loading storage indexes …")
        self._vector_store = VectorStore(
            persist_directory=chroma_dir,
            in_memory=in_memory,
        )
        self._bm25_store = BM25Store(store_path=bm25_dir)

        # ---------------------------------------------------------------- #
        # Station 3 — Retrieval
        # ---------------------------------------------------------------- #
        logger.info("  [3/7] Building hybrid retriever …")
        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            bm25_store=self._bm25_store,
            candidates_per_store=hybrid_candidates,
        )

        logger.info("  [3/7] Loading cross-encoder re-ranker …")
        self._reranker = Reranker()

        # ---------------------------------------------------------------- #
        # Station 5 — Generation (LLM client used also by conflict detector)
        # ---------------------------------------------------------------- #
        logger.info("  [5/7] Initialising LLM client …")
        self._llm_client = FallbackClient(providers=[
                GeminiClient(),
                GroqClient(),
                 # kept as last resort
            ])

        # ---------------------------------------------------------------- #
        # Station 4 — Resolution
        # ---------------------------------------------------------------- #
        logger.info("  [4/7] Wiring truth resolver …")
        self._conflict_detector = ConflictDetector(
            vector_store=self._vector_store,
            llm_callable=self._llm_client,
        )
        self._source_prioritizer = SourcePrioritizer()
        self._truth_resolver = TruthResolver(
            conflict_detector=self._conflict_detector,
            source_prioritizer=self._source_prioritizer,
        )

        # ---------------------------------------------------------------- #
        # Station 5 — Evaluation
        # ---------------------------------------------------------------- #
        logger.info("  [5/7] Configuring confidence scorer …")
        self._confidence_scorer = ConfidenceScorer(
            abstention_threshold=abstention_threshold,
        )

        # ---------------------------------------------------------------- #
        # Station 5 — Prompt builder
        # ---------------------------------------------------------------- #
        logger.info("  [6/7] Configuring prompt builder …")
        self._prompt_builder = PromptBuilder()

        elapsed = time.perf_counter() - t_start
        logger.info(
            "RAG pipeline ready in %.2fs. "
            "Vector store: %d chunk(s). BM25: %d chunk(s).",
            elapsed,
            self._vector_store.count(),
            self._bm25_store.count(),
        )

    # ------------------------------------------------------------------
    # Station 1 — Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_paths: list[str | Path] | None = None,
        json_paths: list[str | Path] | None = None,
        markdown_paths: list[str | Path] | None = None,
        json_template: str | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Run the ingestion pipeline and populate both storage indexes.

        Calls :func:`~ingestion.run_ingestion_pipeline`, then writes all
        deduplicated chunks to both the vector store and BM25 index.

        When ``metadatas`` is not provided, a default metadata dict is
        generated for each chunk containing only a ``source_tier`` of 1.
        For production use, always supply explicit metadata so that tier
        filtering and citation strings are accurate.

        Args:
            pdf_paths: Paths to ``.pdf`` source files.
            json_paths: Paths to ``.json`` or ``.csv`` source files.
            markdown_paths: Paths to ``.md`` or ``.markdown`` files.
            json_template: Template string for JSON/CSV record rendering.
            metadatas: Optional list of metadata dicts, one per ingested
                chunk (after deduplication).  If provided, its length
                must match the number of chunks returned by the ingestion
                pipeline.

        Returns:
            The number of chunks successfully stored in both indexes.

        Raises:
            ValueError: If ``metadatas`` is provided but its length does
                not match the ingested chunk count.
        """
        logger.info("─── Station 1: Ingestion ───")
        t = time.perf_counter()

        result = run_ingestion_pipeline(
            pdf_paths=pdf_paths or [],
            json_paths=json_paths or [],
            markdown_paths=markdown_paths or [],
            json_template=json_template or INCIDENT_TEMPLATE,
        )

        chunks = result.chunks
        if not chunks:
            logger.warning("Ingestion produced no chunks. Indexes not updated.")
            return 0

        if metadatas is not None:
            if len(metadatas) != len(chunks):
                raise ValueError(
                    f"metadatas length ({len(metadatas)}) does not match "
                    f"chunk count ({len(chunks)})."
                )
            resolved_metadatas = metadatas
        else:
            # Build metadata from the actual source files that were ingested.
            # Each chunk gets the name of the file it came from.
            resolved_metadatas = _build_default_metadatas(
                chunks=chunks,
                pdf_paths=pdf_paths or [],
                json_paths=json_paths or [],
                markdown_paths=markdown_paths or [],
    )

        self._vector_store.add_chunks(chunks, resolved_metadatas)
        self._bm25_store.add_chunks(chunks, resolved_metadatas)

        elapsed = time.perf_counter() - t
        logger.info(
            "Ingestion complete in %.2fs: %d chunk(s) stored "
            "(%d duplicate(s) removed, %d source(s) failed).",
            elapsed,
            len(chunks),
            result.duplicates_removed,
            len(result.failed_sources),
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        source_tiers: list[int] | None = None,
    ) -> FinalAnswer:
        """Run the full retrieval-to-generation pipeline for a query.

        Executes all downstream stations (2–7) in sequence.  The
        confidence gate at Station 5 may short-circuit the pipeline
        before the LLM is called, returning a structured abstention
        response.

        Args:
            query_text: The user's natural-language query string.
            source_tiers: Optional per-query tier filter override.
                Defaults to the instance-level ``source_tiers_filter``.

        Returns:
            A fully populated :class:`FinalAnswer` dataclass.  Always
            returns — never raises from pipeline failures.  LLM client
            failures are captured in ``answer`` as a graceful error
            string.

        Raises:
            ValueError: If ``query_text`` is empty.
        """
        if not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")

        tiers = source_tiers or self._source_tiers_filter
        t_query_start = time.perf_counter()
        logger.info("═══ New Query ═══  '%s …'", query_text[:60])

        # ---------------------------------------------------------------- #
        # Station 2+3-A: Hybrid retrieval
        # ---------------------------------------------------------------- #
        logger.info("─── Station 2+3-A: Hybrid Retrieval ───")
        t = time.perf_counter()

        candidates = self._retriever.retrieve(
            query=query_text,
            top_k=self._hybrid_candidates,
            source_tiers=tiers,
        )

        logger.info(
            "Hybrid retrieval: %d candidate(s) in %.2fs.",
            len(candidates), time.perf_counter() - t,
        )

        if not candidates:
            return self._abstention_response(
                query_text,
                reason="Retrieval returned no candidates for this query.",
                elapsed=time.perf_counter() - t_query_start,
            )

        # ---------------------------------------------------------------- #
        # Station 3-B: Cross-encoder re-ranking
        # ---------------------------------------------------------------- #
        logger.info("─── Station 3-B: Re-ranking ───")
        t = time.perf_counter()

        reranked = self._reranker.rerank(
            query=query_text,
            candidates=candidates,
            top_k=self._reranker_top_k,
        )

        logger.info(
            "Re-ranking: top %d chunk(s) selected in %.2fs.",
            len(reranked), time.perf_counter() - t,
        )

        # ---------------------------------------------------------------- #
        # Station 4: Truth resolution
        # ---------------------------------------------------------------- #
        logger.info("─── Station 4: Truth Resolution ───")
        t = time.perf_counter()

        resolution: ResolutionResult = self._truth_resolver.resolve(reranked)

        logger.info(
            "Resolution: %d final chunk(s), %d conflict(s), "
            "%d suppressed in %.2fs.",
            len(resolution.final_chunks),
            resolution.conflict_count,
            resolution.suppressed_count,
            time.perf_counter() - t,
        )

        if not resolution.final_chunks:
            return self._abstention_response(
                query_text,
                reason=(
                    "All retrieved chunks were suppressed during conflict "
                    "resolution. No authoritative content remains."
                ),
                elapsed=time.perf_counter() - t_query_start,
            )

        # ---------------------------------------------------------------- #
        # Station 5: Confidence scoring — the hallucination firewall
        # ---------------------------------------------------------------- #
        logger.info("─── Station 5: Confidence Scoring ───")
        t = time.perf_counter()

        confidence: ConfidenceResult = self._confidence_scorer.should_abstain(
            resolution.final_chunks
        )

        logger.info(
            "Confidence: score=%.3f, abstain=%s in %.2fs.",
            confidence.score, confidence.abstain, time.perf_counter() - t,
        )

        if confidence.abstain:
            abstention_answer = (
                "I don't know based on the available sources.\n\n"
                f"{confidence.explanation}"
            )
            return FinalAnswer(
                query=query_text,
                answer=abstention_answer,
                confidence=confidence,
                conflict_log=resolution.format_log(),
                abstained=True,
                elapsed_seconds=time.perf_counter() - t_query_start,
            )

        # ---------------------------------------------------------------- #
        # Station 6: Prompt construction
        # ---------------------------------------------------------------- #
        logger.info("─── Station 6: Prompt Building ───")
        t = time.perf_counter()

        built: BuiltPrompt = self._prompt_builder.build(
            query=query_text,
            final_chunks=resolution.final_chunks,
            suppressed_chunks=resolution.suppressed_chunks,
        )

        logger.info(
            "Prompt built: %d char(s) user prompt, %d citation(s) in %.2fs.",
            len(built.user_prompt), len(built.citations),
            time.perf_counter() - t,
        )

        # ---------------------------------------------------------------- #
        # Station 7: LLM generation
        # ---------------------------------------------------------------- #
        logger.info("─── Station 7: LLM Generation ───")
        t = time.perf_counter()

        answer_text = self._llm_client(
            prompt=built.user_prompt,
            system_prompt=built.system_prompt,
        )

        logger.info(
            "LLM generation complete in %.2fs. "
            "Session tokens: %s.",
            time.perf_counter() - t,
            self._llm_client.usage,
        )

        # ---------------------------------------------------------------- #
        # Assemble and return the final answer.
        # ---------------------------------------------------------------- #
        total_elapsed = time.perf_counter() - t_query_start
        logger.info("Query complete in %.2fs.", total_elapsed)

        return FinalAnswer(
            query=query_text,
            answer=answer_text,
            confidence=confidence,
            citations=built.citations,
            conflict_log=resolution.format_log(),
            abstained=False,
            elapsed_seconds=total_elapsed,
            source_map=built.source_map,
            has_conflicts=resolution.conflict_count > 0
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _abstention_response(
        self,
        query_text: str,
        reason: str,
        elapsed: float,
    ) -> FinalAnswer:
        """Build a pre-LLM abstention response for pipeline short-circuits.

        Used when retrieval or resolution produces no usable chunks,
        before the confidence scorer is even reached.

        Args:
            query_text: The original query string.
            reason: Human-readable explanation for why the system
                is abstaining at this stage.
            elapsed: Wall-clock seconds elapsed so far.

        Returns:
            A :class:`FinalAnswer` with ``abstained=True`` and a
            zero-value :class:`~evaluation.ConfidenceResult`.
        """
        from evaluation.confidence_scorer import SignalBreakdown

        logger.warning("Pre-confidence abstention: %s", reason)

        zero_signals = SignalBreakdown(
            rerank_signal=0.0,
            tier_signal=0.0,
            agreement_signal=0.0,
        )
        zero_confidence = ConfidenceResult(
            score=0.0,
            signals=zero_signals,
            abstain=True,
            explanation=reason,
            threshold=self._confidence_scorer.abstention_threshold,
        )
        return FinalAnswer(
            query=query_text,
            answer=f"I don't know based on the available sources.\n\n{reason}",
            confidence=zero_confidence,
            abstained=True,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — ingest documents and answer queries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents and run a query:
  python main.py --ingest --query "How long does Procedure X take?"

  # Query only (documents already ingested):
  python main.py --query "What is ERR_CODE_0x4F2?"

  # Ingest only (no query):
  python main.py --ingest
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Natural-language query to run against the ingested documents.",
    )
    parser.add_argument(
        "--ingest", "-i",
        action="store_true",
        help="Run the ingestion pipeline before querying.",
    )
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=[],
        metavar="PATH",
        help="PDF file path(s) to ingest.",
    )
    parser.add_argument(
        "--json",
        nargs="*",
        default=[],
        metavar="PATH",
        help="JSON or CSV file path(s) to ingest.",
    )
    parser.add_argument(
        "--markdown",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Markdown file path(s) to ingest.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=ABSTENTION_THRESHOLD,
        help=f"Confidence abstention threshold (default: {ABSTENTION_THRESHOLD}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RERANKER_TOP_K,
        help=f"Number of chunks shown to the LLM (default: {RERANKER_TOP_K}).",
    )
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Disable source tier filtering (include tier-3 wiki content).",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use in-memory ChromaDB (data not persisted; for testing).",
    )
    return parser


def main() -> None:
    """Entry point for CLI execution.

    Parses arguments, initialises the RAG system, optionally runs
    ingestion, and optionally runs a query, printing the formatted
    result to stdout.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        return

    tiers = None if args.all_tiers else SOURCE_TIERS_FILTER

    # ------------------------------------------------------------------ #
    # Initialise the RAG system (loads all models).
    # ------------------------------------------------------------------ #
    system = RAGSystem(
        abstention_threshold=args.threshold,
        source_tiers_filter=tiers,
        reranker_top_k=args.top_k,
        in_memory=args.in_memory,
    )

    # ------------------------------------------------------------------ #
    # Optional ingestion pass.
    # ------------------------------------------------------------------ #
    if args.ingest:
        chunk_count = system.ingest(
            pdf_paths=args.pdf,
            json_paths=args.json,
            markdown_paths=args.markdown,
        )
        if chunk_count == 0:
            logger.warning(
                "No chunks were ingested. "
                "Check that source file paths are correct."
            )

    # ------------------------------------------------------------------ #
    # Optional query pass.
    # ------------------------------------------------------------------ #
    if args.query:
        answer: FinalAnswer = system.query(args.query)
        print(answer.display())


if __name__ == "__main__":
    main()