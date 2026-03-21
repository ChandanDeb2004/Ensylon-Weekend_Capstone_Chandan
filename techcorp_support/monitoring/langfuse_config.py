"""
langfuse_config.py — pinned to langfuse==2.36.2

langfuse 2.36.2 stable API:
  from langfuse import Langfuse
  lf    = Langfuse(public_key, secret_key, host)
  trace = lf.trace(name, session_id, metadata)   -> StatefulTraceClient
  span  = trace.span(name, metadata)             -> StatefulSpanClient
  event = trace.event(name, metadata, level)     -> logs event on trace
  gen   = trace.generation(name, ...)            -> LLM generation
  lf.flush()                                     -> sends all pending

  from langfuse.callback import CallbackHandler  -> LangChain callback (2.x)

Install:
  pip uninstall langfuse -y
  pip install langfuse==2.36.2
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = None
_initialized = False


def get_langfuse_client():
    """
    Returns initialized Langfuse 2.36.2 client singleton.
    """
    global _client, _initialized

    if _initialized:
        return _client

    _initialized = True

    if not is_langfuse_enabled():
        logger.info("[LANGFUSE] Credentials not configured — tracing disabled.")
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        ok = _client.auth_check()
        if ok:
            logger.info("[LANGFUSE] ✅ Connected to Langfuse Cloud (v2.36.2) — auth OK")
        else:
            logger.warning("[LANGFUSE] ⚠️  auth_check() = False — check your API keys")
        return _client
    except Exception as e:
        logger.warning(f"[LANGFUSE] Client init failed: {e}")
        _client = None
        return None


def get_langfuse_handler(session_id: str = "default", user_id: str = "support-system"):
    """
    LangChain CallbackHandler for automatic LLM call tracing.
    Attaches to CrewAI's LLM so every model call is traced.
    """
    if not is_langfuse_enabled():
        return None
    try:
        from langfuse.callback import CallbackHandler
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
    except Exception as e:
        logger.warning(f"[LANGFUSE] CallbackHandler failed: {e}")
        return None


# Compatibility aliases used in entry points
def setup_langfuse_otel():
    get_langfuse_client()

def configure_langfuse():
    get_langfuse_client()


def flush():
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


def reset():
    global _client, _initialized
    _client = None
    _initialized = False
