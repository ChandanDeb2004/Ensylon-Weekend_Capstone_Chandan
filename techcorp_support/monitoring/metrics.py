"""
metrics.py — langfuse==4.0.1 compatible.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

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
