"""
shared_context.py
Thread-safe shared context cache for cross-agent data access.
"""

import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SharedContext:
    """
    A simple key-value store shared across all agents in a single crew run.
    Thread-safe for concurrent agent access.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, agent: str = "system"):
        with self._lock:
            self._data[key] = value
            logger.debug(f"[CONTEXT] set key='{key}' by agent='{agent}'")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def get_all(self) -> dict:
        with self._lock:
            return dict(self._data)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def clear(self):
        with self._lock:
            self._data.clear()

    def summary(self) -> str:
        with self._lock:
            keys = list(self._data.keys())
            return f"SharedContext({len(keys)} keys): {keys}"


# Module-level singleton — reset per query run
_context = SharedContext()


def get_shared_context() -> SharedContext:
    return _context


def reset_shared_context():
    global _context
    _context = SharedContext()
