"""
escalation_agent.py
Escalation Agent — determines if human intervention is needed and creates tickets.

Fixes applied:
 - Explicit instruction: do NOT call any tool when resolving automatically
 - Tightened expected_output to suppress Thought/Action bleed-through
 - Cleaner output format enforcement
"""

import logging
from crewai import Agent, Task
from agents.base_agent import build_agent
from tools.escalation_tools import (
    create_escalation_ticket, get_escalation_routing,
    notify_support_team, log_escalation_reason,
)

logger = logging.getLogger(__name__)

ESCALATION_AGENT_ROLE = "Escalation Decision and Ticket Management Specialist"

ESCALATION_AGENT_GOAL = """
Apply TechCorp's escalation criteria rigorously to determine whether human intervention
is required. When escalation IS warranted:
1. Call get_escalation_routing(issue_type) to find the right team
2. Call create_escalation_ticket(reason, priority, context) with full context
3. Call notify_support_team(ticket_id) to alert the team
4. Call log_escalation_reason(ticket_id, detailed_reason) with your reasoning

When escalation is NOT warranted:
- Do NOT call any escalation tools
- Simply state your decision and reasoning clearly

ESCALATION TRIGGERS (must escalate if ANY apply):
- Confirmed SLA breach on a guaranteed SLA contract → CRITICAL
- Revenue impact > $100/day reported by customer → HIGH
- Account suspended and customer disputes it → HIGH
- Customer waited > 2 business days without response → HIGH
- Bug confirmed affecting production system → HIGH
- Contradictory data that cannot be resolved automatically → MEDIUM
- Customer explicitly requests human agent → MEDIUM
- Onboarding scenario with >10 users needing setup → MEDIUM
"""

ESCALATION_AGENT_BACKSTORY = """
You are TechCorp's escalation specialist — the final decision-maker on whether a case
needs human eyes. You've handled thousands of support cases and have a finely tuned sense
of when automation has reached its limits.

CRITICAL RULE: When you decide NOT to escalate, you output your decision immediately
without calling any tools. You only call tools (get_escalation_routing,
create_escalation_ticket, notify_support_team, log_escalation_reason) when you have
determined escalation IS needed.

You document every decision with clear reasoning so the human agent (or the system log)
has full context. Your final output is always the structured ESCALATION_FINDINGS block
with no trailing "Action: N/A" or "Thought:" text.
"""


def create_escalation_agent(session_id: str = "default") -> Agent:
    return build_agent(
        role=ESCALATION_AGENT_ROLE,
        goal=ESCALATION_AGENT_GOAL,
        backstory=ESCALATION_AGENT_BACKSTORY,
        tools=[
            create_escalation_ticket,
            get_escalation_routing,
            notify_support_team,
            log_escalation_reason,
        ],
        session_id=session_id,
        max_iter=6,
    )


def create_escalation_task(
    agent: Agent,
    query: str,
    context_str: str,
    force_escalate: bool = False,
    issue_type_hint: str = "",
    context_tasks: list = None,
) -> Task:
    force_note = (
        "NOTE: The Orchestrator has determined that escalation IS required. "
        "Execute the full escalation workflow: routing → ticket → notify → log."
        if force_escalate
        else
        "Apply escalation criteria objectively. If NO triggers are met, output your "
        "decision immediately WITHOUT calling any tools."
    )

    return Task(
        description=f"""
You are the final agent in this investigation. Evaluate whether this case requires human escalation.

QUERY: {query}

SHARED INVESTIGATION CONTEXT (findings from all other agents):
{context_str}

ISSUE TYPE HINT: {issue_type_hint or 'determine from context'}

{force_note}

DECISION RULES:
- If ANY escalation trigger is met → call tools (routing, ticket, notify, log_reason) then output findings
- If NO triggers are met → output findings IMMEDIATELY, call NO tools

TOOLS MAY ONLY BE CALLED WHEN ESCALATING. Do not call log_escalation_reason with ticket_id "N/A".
When escalating: use the EXACT ticket_id string returned by create_escalation_ticket — never write 'ESC-XXXXX'.

Your final output MUST be ONLY the structured block below — no "Thought:", no "Action:", no extra text:

ESCALATION_FINDINGS:
- Escalation Decision: [ESCALATE or RESOLVE_AUTOMATICALLY]
- Priority: [critical/high/medium/low/N/A]
- Issue Type: [sla_violation/billing_dispute/bug_report/onboarding/general/N/A]
- Assigned Team: [team name or N/A]
- Ticket ID: [ESC-XXXXX or N/A]
- Escalation Triggers Met: [list each trigger that applied, or 'none']
- Reasoning: [one clear paragraph explaining your decision]
- Next Steps for Customer: [what the customer should expect next]
""",
        agent=agent,
        context=context_tasks if context_tasks else None,
        expected_output=(
            "Only the ESCALATION_FINDINGS block. "
            "No Thought text, no Action text, no preamble. "
            "Start directly with 'ESCALATION_FINDINGS:' and fill in all fields."
        ),
    )
