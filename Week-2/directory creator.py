'''from pathlib import Path

# Root project directory
ROOT = Path("truth_engine")

# Directory structure
dirs = [
    ROOT,
    ROOT / "ingestion",
    ROOT / "storage",
    ROOT / "retrieval",
    ROOT / "resolution",
    ROOT / "generation",
    ROOT / "evaluation",
]

# Files to create
files = {
    ROOT / "main.py": "# Entrypoint — CLI & orchestration\n",
    ROOT / "config.py": "# Configuration constants\n",
    ROOT / "requirements.txt": "",
    ROOT / "README.md": "# Failure & Mitigation Report\n",

    ROOT / "ingestion" / "__init__.py": "",
    ROOT / "ingestion" / "pdf_ingestor.py": "# PDF + table parsing\n",
    ROOT / "ingestion" / "json_ingestor.py": "# JSON/CSV support log parsing\n",
    ROOT / "ingestion" / "markdown_ingestor.py": "# Header-aware Markdown chunking\n",
    ROOT / "ingestion" / "ingestion_pipeline.py": "# Orchestrates all three ingestors\n",

    ROOT / "storage" / "__init__.py": "",
    ROOT / "storage" / "vector_store.py": "# ChromaDB init, upsert, query\n",
    ROOT / "storage" / "bm25_store.py": "# BM25 index build & keyword search\n",

    ROOT / "retrieval" / "__init__.py": "",
    ROOT / "retrieval" / "hybrid_retriever.py": "# RRF fusion of semantic + BM25\n",
    ROOT / "retrieval" / "reranker.py": "# Cross-encoder re-ranking logic\n",

    ROOT / "resolution" / "__init__.py": "",
    ROOT / "resolution" / "conflict_detector.py": "# Detects contradictions between chunks\n",
    ROOT / "resolution" / "source_prioritizer.py": "# Applies source tier ranking rules\n",
    ROOT / "resolution" / "truth_resolver.py": "# Orchestrates detection + prioritization\n",

    ROOT / "generation" / "__init__.py": "",
    ROOT / "generation" / "prompt_builder.py": "# Builds final prompt with context + citations\n",
    ROOT / "generation" / "llm_client.py": "# LLM API wrapper with error handling\n",

    ROOT / "evaluation" / "__init__.py": "",
    ROOT / "evaluation" / "confidence_scorer.py": "# Computes & returns confidence scores\n",
}

def create_structure():
    # Create directories
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create files
    for file_path, content in files.items():
        if not file_path.exists():
            file_path.write_text(content)

    print("Project structure created successfully.")

if __name__ == "__main__":
    create_structure()
    '''

import anthropic

import os

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


print(os.getenv("ANTHROPIC_API_KEY"))
