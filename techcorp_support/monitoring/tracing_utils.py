"""
tracing_utils.py — langfuse==4.0.1

Langfuse v4 auto-traces all LLM calls via OTEL when configured.
This module handles only BUSINESS-LEVEL events that OTEL cannot detect:
  - escalation triggered
  - conflict detected
  - SLA breach confirmed
  - tool failures
  - routing decisions

All events attach to the current active trace span automatically.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def log_decision_point(
    session_id: str,
    decision: str,
    reasoning: str,
    agent: str = "orchestrator",
    metadata: Optional[dict] = None,
):
    logger.info(f"[DECISION] {agent} → {decision}")
    from monitoring.langfuse_config import log_event
    log_event("decision_point", {
        "agent":     agent,
        "decision":  decision,
        "reasoning": reasoning[:300],
        "session_id": session_id,
        **(metadata or {}),
    })


def log_conflict(session_id: str, conflict_description: str, resolution: str = ""):
    logger.warning(f"[CONFLICT] {conflict_description[:120]}")
    from monitoring.langfuse_config import log_event
    log_event("conflict_detected", {
        "conflict":   conflict_description[:300],
        "resolution": resolution[:300],
        "session_id": session_id,
    })


def log_escalation_event(session_id: str, ticket_id: str, priority: str, reason: str):
    logger.info(f"[ESCALATION] ticket={ticket_id} | priority={priority}")
    from monitoring.langfuse_config import log_event
    log_event("escalation_triggered", {
        "ticket_id":  ticket_id,
        "priority":   priority,
        "reason":     reason[:300],
        "session_id": session_id,
    })


def log_tool_failure(session_id: str, tool_name: str, error: str):
    logger.error(f"[TOOL FAILURE] {tool_name}: {error[:100]}")
    from monitoring.langfuse_config import log_event
    log_event("tool_failure", {
        "tool":       tool_name,
        "error":      error[:300],
        "session_id": session_id,
    })
