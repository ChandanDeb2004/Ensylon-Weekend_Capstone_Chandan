"""
escalation_agent.py
Escalation Agent — determines if human intervention is needed and creates tickets.

Groq fix: The model was writing ESCALATION_FINDINGS block AND calling a tool
in the same response — Groq rejects mixed text+tool_call responses.
Fix: Two-phase approach — tools first, THEN output block. Never simultaneously.
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
Determine whether a support case requires human escalation and act accordingly.

STRICT TWO-PHASE PROCESS — never mix tool calls with text output:
PHASE 1 (tool calls only): If escalating, call tools one at a time. No text output yet.
PHASE 2 (text output only): After ALL tools are done, write the ESCALATION_FINDINGS block.

When escalation is NOT warranted: skip Phase 1 entirely, go straight to Phase 2.

ESCALATION TRIGGERS:
- Confirmed SLA breach on guaranteed contract → CRITICAL
- Revenue impact > $100/day → HIGH
- Account suspended and disputed → HIGH
- Waited > 2 business days → HIGH
- Confirmed production bug → HIGH
- Unresolvable contradiction → MEDIUM
- Customer requests human → MEDIUM
- Onboarding > 10 users → MEDIUM
"""

ESCALATION_AGENT_BACKSTORY = """
You are TechCorp's escalation specialist. You follow a strict two-phase process:

Phase 1 — Tool calls (if escalating):
  Call get_escalation_routing first.
  Then call create_escalation_ticket.
  Then call notify_support_team.
  Then call log_escalation_reason.
  Each tool call is a SEPARATE step. Never call a tool and write text in the same step.

Phase 2 — Write ESCALATION_FINDINGS block (always last, always text only):
  After all tools are done (or immediately if not escalating), write the findings block.
  Never call a tool while writing the findings block.
  Never write the findings block while calling a tool.
"""


def create_escalation_agent(session_id: str = "default") -> Agent:
    return build_agent(
        role=ESCALATION_AGENT_ROLE,
        goal=ESCALATION_AGENT_GOAL,
        backstory=ESCALATION_AGENT_BACKSTORY,
        tools=[
            get_escalation_routing,
            create_escalation_ticket,
            notify_support_team,
            log_escalation_reason,
        ],
        session_id=session_id,
        max_iter=8,
    )


def create_escalation_task(
    agent: Agent,
    query: str,
    context_str: str,
    force_escalate: bool = False,
    issue_type_hint: str = "",
    context_tasks: list = None,
) -> Task:

    if force_escalate:
        workflow = """
⚠️  ESCALATION IS REQUIRED. Follow this exact sequence:
STEP 1 (tool call): get_escalation_routing("{issue_type}")
STEP 2 (tool call): create_escalation_ticket(reason, priority, context)
STEP 3 (tool call): notify_support_team(ticket_id_from_step_2)
STEP 4 (tool call): log_escalation_reason(ticket_id_from_step_2, reasoning)
STEP 5 (text only): Write ESCALATION_FINDINGS block using ticket_id from Step 2.
""".format(issue_type=issue_type_hint or "general")
    else:
        workflow = """
DECISION PROCESS:
A) Check if any escalation trigger applies.
B) If YES → follow Steps 1-5 above (tools first, findings block last).
C) If NO  → skip all tools, go directly to writing ESCALATION_FINDINGS block.

⚠️  NEVER write ESCALATION_FINDINGS and call a tool in the same response step.
    Tools first. Text output last. Never together.
"""

    return Task(
        description=f"""
Evaluate whether this support case requires human escalation.

QUERY: {query}

SHARED INVESTIGATION CONTEXT:
{context_str}

ISSUE TYPE HINT: {issue_type_hint or 'determine from context'}

{workflow}

ESCALATION TRIGGERS (escalate if ANY apply):
- Confirmed SLA breach on guaranteed contract → CRITICAL
- Revenue impact > $100/day → HIGH
- Account suspended and disputed → HIGH
- Waited > 2 business days without response → HIGH
- Confirmed production bug → HIGH
- Unresolvable contradiction → MEDIUM
- Customer requests human agent → MEDIUM
- Onboarding > 10 users → MEDIUM

FINAL OUTPUT — write this block ONLY after all tool calls are complete:

ESCALATION_FINDINGS:
- Escalation Decision: [ESCALATE or RESOLVE_AUTOMATICALLY]
- Priority: [critical/high/medium/low/N/A]
- Issue Type: [sla_violation/billing_dispute/bug_report/onboarding/general/N/A]
- Assigned Team: [exact value from get_escalation_routing, or N/A]
- Ticket ID: [exact ticket_id from create_escalation_ticket, or N/A]
- Escalation Triggers Met: [list or 'none']
- Reasoning: [one clear paragraph]
- Next Steps for Customer: [what happens next]
""",
        agent=agent,
        context=context_tasks if context_tasks else None,
        expected_output=(
            "Only the ESCALATION_FINDINGS block with all fields filled. "
            "No Thought text, no Action text, no tool call syntax, no preamble. "
            "Start with 'ESCALATION_FINDINGS:' on the first line."
        ),
    )
