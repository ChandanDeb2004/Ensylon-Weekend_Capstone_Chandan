"""
metrics.py — langfuse==4.0.1 compatible.
"""
import logging
from datetime import datetime, timezone
from typing import Optional
import os
logger = logging.getLogger(__name__)
_run_metrics: list[dict] = []


def record_run_metrics(
    session_id: str,
    query: str,
    agents_used: list[str],
    total_duration_s: float,
    escalated: bool,
    tool_failures: int,
    conflicts_found: int,
    resolution: str,
) -> dict:
    record = {
        "session_id":         session_id,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "query_preview":      query[:80],
        "agents_used":        agents_used,
        "agent_count":        len(agents_used),
        "total_duration_s":   round(total_duration_s, 2),
        "escalated":          escalated,
        "tool_failures":      tool_failures,
        "conflicts_found":    conflicts_found,
        "resolution_preview": resolution[:200],
    }
    _run_metrics.append(record)

    try:
        from monitoring.tracing_utils import _safe_log_event
        _safe_log_event(name="run_metrics", session_id=session_id, metadata=record)
    except Exception as e:
        logger.debug(f"[METRICS] Langfuse push skipped: {e}")

    logger.info(
        f"[METRICS] session={session_id} | agents={len(agents_used)} | "
        f"duration={total_duration_s:.1f}s | escalated={escalated} | failures={tool_failures}"
    )
    return record


def get_all_metrics() -> list[dict]:
    return list(_run_metrics)


def get_session_metrics(session_id: str) -> Optional[dict]:
    for m in reversed(_run_metrics):
        if m["session_id"] == session_id:
            return m
    return None

# monitoring/metrics.py — add this function

def record_token_usage(crew_output, session_id: str):
    """
    Extract token usage from CrewAI crew output and log to Langfuse.
    Call this right after crew.kickoff() in orchestrator.py
    """
    usage = {
        "session_id":    session_id,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "total_tokens":      0,
        "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
    }

    # CrewAI exposes token usage on the output object
    try:
        if hasattr(crew_output, "token_usage"):
            tu = crew_output.token_usage
            usage["prompt_tokens"]     = getattr(tu, "prompt_tokens",     0)
            usage["completion_tokens"] = getattr(tu, "completion_tokens", 0)
            usage["total_tokens"]      = getattr(tu, "total_tokens",      0)
    except Exception:
        pass

    logger.info(
        f"[TOKENS] session={session_id} | "
        f"prompt={usage['prompt_tokens']} | "
        f"completion={usage['completion_tokens']} | "
        f"total={usage['total_tokens']}"
    )

    # Push to Langfuse
    try:
        from monitoring.tracing_utils import _safe_log_event
        _safe_log_event(name="token_usage", session_id=session_id, metadata=usage)
    except Exception:
        pass

    return usage