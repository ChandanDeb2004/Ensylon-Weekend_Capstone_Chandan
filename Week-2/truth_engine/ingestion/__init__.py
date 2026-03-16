"""
ingestion/
~~~~~~~~~~

Station 1 of the RAG pipeline: raw document ingestion.

Reads PDF, JSON/CSV, and Markdown source files and outputs a single
deduplicated list of plain-text chunks ready for embedding.

"""

from ingestion.ingestion_pipeline import IngestionResult, run_ingestion_pipeline
from ingestion.json_ingestor import ingest_json
from ingestion.markdown_ingestor import ingest_markdown
from ingestion.pdf_ingestor import ingest_pdf

__all__ = [
    "run_ingestion_pipeline",
    "IngestionResult",
    "ingest_pdf",
    "ingest_json",
    "ingest_markdown",
]