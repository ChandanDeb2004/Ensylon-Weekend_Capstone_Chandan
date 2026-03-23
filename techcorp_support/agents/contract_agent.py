"""
contract_agent.py
Contract Agent — retrieves contracts, validates SLAs, checks entitlements.
"""

import logging
from datetime import datetime, timedelta, timezone
from crewai import Agent, Task
from agents.base_agent import build_agent
from tools.contract_tools import (
    lookup_contract, get_contract_terms,
    validate_sla_compliance, get_included_features,
)

logger = logging.getLogger(__name__)

CONTRACT_AGENT_ROLE = "Contract and SLA Compliance Analyst"

CONTRACT_AGENT_GOAL = """
Review customer contracts rigorously and determine:
- Exact SLA terms (response times, SLA type: best-effort vs guaranteed)
- Whether an SLA breach has occurred (calculate hours elapsed vs guaranteed)
- Feature entitlements as written in the contract (the legal source of truth)
- Any special terms, pricing addenda, or custom agreements
- Compensation clauses triggered by SLA violations

When there is a conflict between what documentation says and what the contract specifies,
the CONTRACT IS ALWAYS THE AUTHORITATIVE SOURCE. Flag such conflicts explicitly.

Provide a clear breach/no-breach determination with supporting calculations when SLA
validation is requested.
"""

CONTRACT_AGENT_BACKSTORY = """
You are TechCorp's contract compliance analyst with a background in enterprise SaaS legal
agreements. You have reviewed hundreds of contracts and know that the details in contracts
often differ from what sales teams promise verbally or what public documentation states.

You are methodical with dates and time calculations — when checking SLA compliance, you
always calculate (hours elapsed) vs (SLA guarantee hours) and provide the exact delta.
You never give ambiguous answers: a breach either occurred or it didn't.

You are also skilled at identifying when a contract's special terms override the standard
plan features — especially API rate limits documented in pricing addenda that contradict
the public feature matrix.
"""


def create_contract_agent(session_id: str = "default") -> Agent:
    return build_agent(
        role=CONTRACT_AGENT_ROLE,
        goal=CONTRACT_AGENT_GOAL,
        backstory=CONTRACT_AGENT_BACKSTORY,
        tools=[
            lookup_contract,
            get_contract_terms,
            validate_sla_compliance,
            get_included_features,
        ],
        session_id=session_id,
        max_iter=8,
    )


def create_contract_task(
    agent: Agent,
    query: str,
    context_str: str,
    contract_id: str = "CTR-001",
    check_sla: bool = False,
    issue_date: str = "",
    context_tasks: list = None,
) -> Task:
    sla_instruction = ""
    if check_sla:
        date_to_use = issue_date or (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        sla_instruction = f"""
            5. PRIORITY: Call validate_sla_compliance({contract_id}, '{date_to_use}') to determine
            if an SLA breach has occurred. Calculate exact hours elapsed vs SLA guarantee.
            Determine breach severity and any compensation clauses that apply.
            """

    return Task(
        description=f"""
            Investigate contract terms and SLA compliance for this support query:

            QUERY: {query}

            SHARED INVESTIGATION CONTEXT:
            {context_str}

            CONTRACT ID: {contract_id}

            ⛔ HARD RULE: Every tool call MUST use "{contract_id}" as the argument.
            Do not use CTR-001, CTR-002 unless {contract_id} matches exactly.
            Use ONLY: {contract_id}

            YOUR TASK — call these tools in order:
            Step 1: lookup_contract("{contract_id}") → read the plan, SLA type, all fields
            Step 2: get_contract_terms("{contract_id}") → get exact SLA hours and special terms
            Step 3: get_included_features("{contract_id}") → list all features in contract
            {sla_instruction}
            Step 5: If any tool fails, document it and fill that field as "tool failure - unavailable"
            Step 6: Fill every field in CONTRACT_FINDINGS using the actual data from Steps 1-4.
                    Do NOT write "unknown" or "to be determined" — use real values from tool responses.

            CONFLICT RESOLUTION RULE: If contract terms conflict with documentation or account
            settings, the contract is the authoritative source. State this explicitly.

            OUTPUT FORMAT — always end with a structured findings block:
            CONTRACT_FINDINGS:
            - Contract ID: [CTR-XXX]
            - Plan in Contract: [plan name]
            - SLA Type: [guaranteed/best-effort]
            - SLA Response Hours: [N hours]
            - SLA Breach Occurred: [yes/no/cannot determine]
            - Hours Overdue (if breach): [N hours or N/A]
            - Compensation Applicable: [describe or 'none']
            - Contractual Feature Entitlements: [list]
            - Special Terms / Addenda: [describe or 'none']
            - Conflicts with Documentation/Account: [describe or 'none']
            - Tool Failures: [list or 'none']
            - Confidence: [high/medium/low]
            """,
                    agent=agent,
                    context=context_tasks if context_tasks else None,
                    expected_output=(
                        "Only the CONTRACT_FINDINGS block. "
                        "No 'Thought:' text, no 'Action:' text, no preamble, no code fences. "
                        "Start directly with 'CONTRACT_FINDINGS:' and fill in every field."
                    ),
                )
