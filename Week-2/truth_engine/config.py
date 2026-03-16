# Configuration constants
SOURCE_TIER_MAP = {"manual": 1, "support_logs": 2, "wiki": 3}
SIMILARITY_THRESHOLD = 0.35       # Below this → "I don't know"
CONFLICT_SIMILARITY_THRESHOLD = 0.85  # Above this → check for conflict
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
CHROMA_PERSIST_DIR = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index.pkl"
LLM_MODEL = "gpt-4o-mini"         # swap for any OpenAI-compatible model

TIER_PENALTIES: dict[int, float] = {
    1: 0.00,
    2: 0.10,
    3: 0.20,
}
DEFAULT_TIER_PENALTY: float = 0.25
ABSTENTION_THRESHOLD: float = 0.40