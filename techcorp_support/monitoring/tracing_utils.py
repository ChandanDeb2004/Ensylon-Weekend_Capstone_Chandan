"""
tracing_utils.py — langfuse==2.36.2

Stable 2.x pattern:
  lf    = Langfuse(...)
  trace = lf.trace(name=..., session_id=..., metadata=...)
  trace.event(name=..., metadata=..., level=...)
  lf.flush()
"""

import logging
import functools
from typing import Callable, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _safe_log_event(
    name: str,
    session_id: str,
    level: str = "DEFAULT",
    metadata: Optional[dict] = None,
):
    """
    Creates a trace and attaches an event to it.
    langfuse 2.36.2: lf.trace() → trace.event() → lf.flush()
    """
    from monitoring.langfuse_config import get_langfuse_client
    lf = get_langfuse_client()

    if lf is None:
        logger.debug(f"[TRACE:{name}] no client | {metadata}")
        return

    try:
        trace = lf.trace(
            name=name,
            session_id=session_id,
            metadata=metadata or {},
        )
        trace.event(
            name=name,
            level=level,
            metadata=metadata or {},
        )
        lf.flush()
    except Exception as e:
        logger.debug(f"[TRACING] '{name}' failed: {e}")


def trace_agent_call(agent_name: str, session_id: str = "default"):
    """Wraps agent execution in a Langfuse trace span."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from monitoring.langfuse_config import get_langfuse_client
            lf = get_langfuse_client()
            trace = None
            span  = None

            if lf:
                try:
                    trace = lf.trace(
                        name=f"agent:{agent_name}",
                        session_id=session_id,
                        metadata={
                            "agent": agent_name,
                            "started_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    span = trace.span(name=f"{agent_name}_execution")
                except Exception as e:
                    logger.debug(f"[TRACING] span init failed for {agent_name}: {e}")

            try:
                result = func(*args, **kwargs)
                if span:
                    try:
                        span.end(output=str(result)[:500])
                        lf.flush()
                    except Exception:
                        pass
                return result
            except Exception as exc:
                if trace:
                    try:
                        trace.event(
                            name="agent_error",
                            level="ERROR",
                            metadata={"agent": agent_name, "error": str(exc)[:300]},
                        )
                        lf.flush()
                    except Exception:
                        pass
                raise

        return wrapper
    return decorator


def log_decision_point(
    session_id: str,
    decision: str,
    reasoning: str,
    agent: str = "orchestrator",
    metadata: Optional[dict] = None,
):
    logger.info(f"[DECISION] {agent} → {decision}")
    _safe_log_event(
        name="decision_point",
        session_id=session_id,
        metadata={
            "agent":     agent,
            "decision":  decision,
            "reasoning": reasoning[:300],
            **(metadata or {}),
        },
    )


def log_conflict(session_id: str, conflict_description: str, resolution: str = ""):
    logger.warning(f"[CONFLICT] {conflict_description[:120]}")
    _safe_log_event(
        name="conflict_detected",
        session_id=session_id,
        level="WARNING",
        metadata={
            "conflict":   conflict_description[:300],
            "resolution": resolution[:300],
        },
    )


def log_escalation_event(session_id: str, ticket_id: str, priority: str, reason: str):
    logger.info(f"[ESCALATION] ticket={ticket_id} | priority={priority}")
    _safe_log_event(
        name="escalation_triggered",
        session_id=session_id,
        level="WARNING",
        metadata={
            "ticket_id": ticket_id,
            "priority":  priority,
            "reason":    reason[:300],
        },
    )


def log_tool_failure(session_id: str, tool_name: str, error: str):
    logger.error(f"[TOOL FAILURE] {tool_name}: {error[:100]}")
    _safe_log_event(
        name="tool_failure",
        session_id=session_id,
        level="ERROR",
        metadata={
            "tool":  tool_name,
            "error": error[:300],
        },
    )
