"""
escalation_tools.py
Mock tools for the Escalation Agent.
Simulates a ticketing/escalation service.
"""

import uuid
import logging
from datetime import datetime, timezone
from crewai.tools import tool
from tools.tool_base import with_tool_behavior, format_tool_result

logger = logging.getLogger(__name__)

# ── In-memory ticket store ─────────────────────────────────────────────────────
_TICKET_STORE: dict = {}

ROUTING_TABLE = {
    "sla_violation":       {"team": "Customer Success",    "channel": "slack:#cs-escalations",  "sla_target": "1 hour"},
    "billing_dispute":     {"team": "Finance",              "channel": "email:billing@techcorp.io","sla_target": "4 hours"},
    "bug_report":          {"team": "Engineering",          "channel": "jira:BUGS",              "sla_target": "2 hours"},
    "account_suspension":  {"team": "Account Management",  "channel": "slack:#am-critical",     "sla_target": "30 minutes"},
    "feature_request":     {"team": "Product",             "channel": "slack:#product-feedback", "sla_target": "24 hours"},
    "onboarding":          {"team": "Onboarding",          "channel": "email:onboard@techcorp.io","sla_target": "2 hours"},
    "general":             {"team": "Support",             "channel": "zendesk:general",         "sla_target": "8 hours"},
    "contract_dispute":    {"team": "Legal & CS",          "channel": "email:contracts@techcorp.io","sla_target": "2 hours"},
}

PRIORITY_LEVELS = {
    "critical": {"label": "P0 - Critical",   "color": "red",    "notify_immediately": True},
    "high":     {"label": "P1 - High",       "color": "orange", "notify_immediately": True},
    "medium":   {"label": "P2 - Medium",     "color": "yellow", "notify_immediately": False},
    "low":      {"label": "P3 - Low",        "color": "green",  "notify_immediately": False},
}


# ── Tool Implementations ───────────────────────────────────────────────────────

@with_tool_behavior("create_escalation_ticket")
def _create_escalation_ticket_impl(reason: str, priority: str, context: str) -> dict:
    ticket_id = f"ESC-{str(uuid.uuid4())[:8].upper()}"
    priority_lower = priority.lower()
    priority_info = PRIORITY_LEVELS.get(priority_lower, PRIORITY_LEVELS["medium"])

    ticket = {
        "ticket_id": ticket_id,
        "reason": reason,
        "priority": priority_lower,
        "priority_label": priority_info["label"],
        "context_summary": context[:500],  # truncate for storage
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assigned_team": None,  # set by routing
        "notify_immediately": priority_info["notify_immediately"],
    }
    _TICKET_STORE[ticket_id] = ticket
    logger.info(f"[ESCALATION] Created ticket {ticket_id} | priority={priority} | reason={reason[:80]}")
    return ticket


@with_tool_behavior("get_escalation_routing")
def _get_escalation_routing_impl(issue_type: str) -> dict:
    key = issue_type.lower().replace(" ", "_")
    route = ROUTING_TABLE.get(key, ROUTING_TABLE["general"])
    return {
        "issue_type": issue_type,
        "assigned_team": route["team"],
        "notification_channel": route["channel"],
        "sla_target": route["sla_target"],
        "routing_note": f"Routing based on issue type '{issue_type}'.",
    }


@with_tool_behavior("notify_support_team")
def _notify_support_team_impl(ticket_id: str) -> dict:
    ticket = _TICKET_STORE.get(ticket_id)
    if not ticket:
        return {"success": False, "message": f"Ticket {ticket_id} not found."}
    notification = {
        "ticket_id": ticket_id,
        "notification_sent": True,
        "notified_at": datetime.now(timezone.utc).isoformat(),
        "channels_alerted": ["email", "slack"],
        "message": f"Team notified about {ticket.get('priority_label', 'ticket')} escalation.",
    }
    _TICKET_STORE[ticket_id]["notification_sent"] = True
    return notification


@with_tool_behavior("log_escalation_reason")
def _log_escalation_reason_impl(ticket_id: str, reason: str) -> dict:
    ticket = _TICKET_STORE.get(ticket_id)
    if not ticket:
        return {"success": False, "message": f"Ticket {ticket_id} not found."}
    _TICKET_STORE[ticket_id]["detailed_reason"] = reason
    _TICKET_STORE[ticket_id]["reason_logged_at"] = datetime.now(timezone.utc).isoformat()
    return {
        "ticket_id": ticket_id,
        "reason_logged": True,
        "reason_summary": reason[:200],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── CrewAI Tool Wrappers ───────────────────────────────────────────────────────

@tool("Create Escalation Ticket")
def create_escalation_ticket(reason: str, priority: str, context: str) -> str:
    """
    Create a formal escalation ticket for human intervention.
    Input: reason (string), priority ('critical'/'high'/'medium'/'low'), context (string summary)
    Returns: ticket ID, priority label, creation timestamp.
    Use when a customer issue requires human agent intervention.
    """
    result = _create_escalation_ticket_impl(reason, priority, context)
    return format_tool_result(result)


@tool("Get Escalation Routing")
def get_escalation_routing(issue_type: str) -> str:
    """
    Determine which team and channel an escalation should be routed to.
    Input: issue_type (string, e.g. 'sla_violation', 'billing_dispute', 'bug_report', 'onboarding')
    Returns: assigned team, notification channel, SLA target for response.
    Use before creating a ticket to determine correct routing.
    """
    result = _get_escalation_routing_impl(issue_type)
    return format_tool_result(result)


@tool("Notify Support Team")
def notify_support_team(ticket_id: str) -> str:
    """
    Send an alert to the support team about a newly created escalation ticket.
    Input: ticket_id (string, from create_escalation_ticket output)
    Returns: confirmation that team was notified, channels used.
    Use after creating a ticket to ensure immediate team awareness.
    """
    result = _notify_support_team_impl(ticket_id)
    return format_tool_result(result)


@tool("Log Escalation Reason")
def log_escalation_reason(ticket_id: str, reason: str) -> str:
    """
    Document the detailed reasoning behind an escalation decision.
    Input: ticket_id (string), reason (detailed string explanation)
    Returns: confirmation of logging.
    Use to attach detailed reasoning to an escalation ticket for the human agent.
    """
    result = _log_escalation_reason_impl(ticket_id, reason)
    return format_tool_result(result)


def get_all_tickets() -> dict:
    """Utility for UI display — not a CrewAI tool."""
    return _TICKET_STORE
