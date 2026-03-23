"""
langfuse_config.py — langfuse==4.0.1
Minimal setup: configure once, then let OTEL auto-instrument everything.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_configured = False


def configure():
    """
    Call once at startup. Langfuse v4 then auto-traces all LLM calls
    via OpenTelemetry — no manual spans needed.
    """
    global _configured
    if _configured:
        return
    if not is_langfuse_enabled():
        logger.info("[LANGFUSE] Credentials not set — tracing disabled.")
        return
    try:
        import langfuse
        langfuse.configure(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        _configured = True
        logger.info("[LANGFUSE] ✅ Configured — OTEL auto-instrumentation active (v4)")
    except Exception as e:
        logger.warning(f"[LANGFUSE] Configure failed: {e}")


def log_event(name: str, metadata: dict = None):
    """
    Log a custom business event (escalation, conflict, SLA breach etc.)
    to the current active Langfuse trace span.
    """
    if not _configured:
        return
    try:
        import langfuse
        client = langfuse.get_client()
        client.create_event(name=name, metadata=metadata or {})
    except Exception as e:
        logger.debug(f"[LANGFUSE] log_event '{name}' skipped: {e}")


def flush():
    """Flush pending spans before process exit."""
    if not _configured:
        return
    try:
        import langfuse
        langfuse.flush()
    except Exception:
        pass


def is_langfuse_enabled() -> bool:
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    return bool(pk and sk and pk.startswith("pk-lf") and sk.startswith("sk-lf"))


def reset():
    """Force re-init — for testing."""
    global _configured
    _configured = False
