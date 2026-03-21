"""
langfuse_config.py
Verified for langfuse==4.0.1

langfuse 4.0.1 actual API:
  from langfuse import Langfuse
  lf = Langfuse(public_key, secret_key, host)  <- constructor still exists
  trace = lf.trace(name, session_id)            <- returns StatefulTraceClient
  trace.event(name, metadata, level)            <- attaches event to trace
  lf.flush()                                    <- flush async queue

  CallbackHandler (for LangChain/CrewAI LLM tracing):
  from langfuse.langchain import CallbackHandler  <- moved in v4 (was langfuse.callback)

  langfuse.configure() -> DOES NOT EXIST in any version, ignore previous attempts
  langfuse.get_client() -> DOES NOT EXIST in 4.0.1
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Manual singleton — never use @lru_cache (caches failed None forever)
_client = None
_initialized = False


def get_langfuse_client():
    """
    Returns initialized Langfuse 4.0.1 client.
    Uses Langfuse() constructor — same as v2, still valid in v4.
    """
    global _client, _initialized

    if _initialized:
        return _client

    _initialized = True

    if not is_langfuse_enabled():
        logger.info("[LANGFUSE] Credentials not configured — tracing disabled.")
        _client = None
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("[LANGFUSE] Client initialized → Langfuse Cloud (v4.0.1)")
        return _client
    except Exception as e:
        logger.warning(f"[LANGFUSE] Client init failed: {e} — tracing disabled.")
        _client = None
        return None


def get_langfuse_handler(session_id: str = "default", user_id: str = "support-system"):
    """
    LangChain CallbackHandler for CrewAI LLM call tracing.
    In langfuse v4 this moved from langfuse.callback -> langfuse.langchain
    """
    if not is_langfuse_enabled():
        return None
    try:
        # v4: moved to langfuse.langchain
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            session_id=session_id,
            user_id=user_id,
            trace_name="techcorp-support-crew",
        )
        logger.info(f"[LANGFUSE] CallbackHandler ready | session={session_id}")
        return handler
    except ImportError:
        # Fallback: try old path for safety
        try:
            from langfuse.callback import CallbackHandler
            return CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                session_id=session_id,
            )
        except Exception:
            logger.warning(f"[LANGFUSE] CallbackHandler fallback failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"[LANGFUSE] CallbackHandler failed: {e}")
        return None


def configure_langfuse():
    """
    Compatibility shim — configure() does not exist in langfuse 4.0.1.
    Simply initializes the client singleton.
    """
    get_langfuse_client()


def flush():
    """Flush pending traces."""
    lf = get_langfuse_client()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass


def is_langfuse_enabled() -> bool:
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    return bool(pk and sk and pk.startswith("pk-lf") and sk.startswith("sk-lf"))


def reset_client():
    """Force re-init — useful for testing."""
    global _client, _initialized
    _client = None
    _initialized = False
