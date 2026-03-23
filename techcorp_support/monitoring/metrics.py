"""
metrics.py — per-run metrics + token usage tracking.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)
_run_metrics: list[dict] = []


def record_token_usage(crew_output, session_id: str) -> dict:
    """Extract token usage from CrewAI crew output."""
    usage = {
        "session_id":        session_id,
        "prompt_tokens":     0,
        "completion_tokens": 0,
        "total_tokens":      0,
    }
    try:
        if hasattr(crew_output, "token_usage") and crew_output.token_usage:
            tu = crew_output.token_usage
            usage["prompt_tokens"]     = getattr(tu, "prompt_tokens",     0) or 0
            usage["completion_tokens"] = getattr(tu, "completion_tokens", 0) or 0
            usage["total_tokens"]      = getattr(tu, "total_tokens",      0) or 0
    except Exception:
        pass

    logger.info(
        f"[TOKENS] session={session_id} | "
        f"prompt={usage['prompt_tokens']} | "
        f"completion={usage['completion_tokens']} | "
        f"total={usage['total_tokens']}"
    )

    # Log to Langfuse as a business event
    from monitoring.langfuse_config import log_event
    log_event("token_usage", usage)

    return usage


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

    from monitoring.langfuse_config import log_event
    log_event("run_metrics", record)

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
