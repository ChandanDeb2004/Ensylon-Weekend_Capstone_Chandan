"""
conversation_memory.py
Persistent conversation memory across multiple turns.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("results/sessions")


class ConversationMemory:
    """
    Persists conversation history to disk so sessions survive restarts.
    Also provides a rolling window view for LLM context injection.
    """

    def __init__(self, session_id: str, max_turns: int = 20):
        self.session_id = session_id
        self.max_turns = max_turns
        self._history: list[dict] = []
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._path = MEMORY_DIR / f"{session_id}.json"
        self._load()

    def add(self, role: str, content: str, metadata: Optional[dict] = None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self._history.append(entry)
        self._prune()
        self._save()

    def get_recent(self, n: int = 10) -> list[dict]:
        return self._history[-n:]

    def format_for_prompt(self, n: int = 8) -> str:
        recent = self.get_recent(n)
        lines = []
        for turn in recent:
            role = turn["role"].upper()
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def _prune(self):
        if len(self._history) > self.max_turns:
            self._history = self._history[:1] + self._history[-(self.max_turns - 1):]

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump({"session_id": self.session_id, "history": self._history}, f, indent=2)
        except Exception as e:
            logger.error(f"[MEMORY] Failed to save session: {e}")

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                    self._history = data.get("history", [])
                    logger.info(f"[MEMORY] Loaded {len(self._history)} turns for session {self.session_id}")
            except Exception as e:
                logger.warning(f"[MEMORY] Could not load session {self.session_id}: {e}")
                self._history = []

    def clear(self):
        self._history = []
        if self._path.exists():
            self._path.unlink()
