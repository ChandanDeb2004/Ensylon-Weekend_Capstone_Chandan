"""
account_agent.py
Account Agent — investigates customer accounts, billing, seats, and plan details.
"""

import logging
from crewai import Agent, Task
from agents.base_agent import build_agent
from tools.account_tools import (
    lookup_customer, get_billing_history,
    check_account_status, list_enabled_features,
)

logger = logging.getLogger(__name__)

ACCOUNT_AGENT_ROLE = "Customer Account Investigator"

ACCOUNT_AGENT_GOAL = """
Investigate the customer's account thoroughly and provide accurate, complete findings about:
- Their current subscription plan and tier
- Seat allocation vs actual usage (flag overages)
- Billing history and payment status
- Features currently enabled on their account
- Account status (active, suspended, overdue)

Always cross-check findings for internal consistency. If data is incomplete due to a tool failure,
explicitly state what could not be verified and suggest a fallback approach.
"""

ACCOUNT_AGENT_BACKSTORY = """
You are a senior account analyst at TechCorp with 8 years of experience investigating
customer accounts. You have deep knowledge of TechCorp's subscription tiers (Starter, Pro,
Enterprise), seat licensing models, and billing systems.

You are methodical and thorough — you never make assumptions about a customer's plan without
verifying it from the database. When tools fail, you document the gap clearly so the
Orchestrator can route around it. You communicate findings in structured, factual language
that other agents can rely on.

You are especially skilled at spotting discrepancies: seat overages, suspended accounts,
billing gaps, and mismatches between what a customer believes their plan includes versus what
is actually configured in the system.
"""


def create_account_agent(session_id: str = "default") -> Agent:
    return build_agent(
        role=ACCOUNT_AGENT_ROLE,
        goal=ACCOUNT_AGENT_GOAL,
        backstory=ACCOUNT_AGENT_BACKSTORY,
        tools=[
            lookup_customer,
            get_billing_history,
            check_account_status,
            list_enabled_features,
        ],
        session_id=session_id,
        max_iter=8,
    )


def create_account_task(agent: Agent, query: str, context_str: str, customer_id: str = "CUST-001") -> Task:
    return Task(
        description=f"""
Investigate the customer account for this support query:

QUERY: {query}

SHARED INVESTIGATION CONTEXT:
{context_str}

CUSTOMER ID: {customer_id}

⛔ HARD RULE: Every single tool call in this task MUST use "{customer_id}" as the argument.
   Using ANY other customer ID (CUST-001, CUST-002, etc.) is WRONG unless {customer_id} matches.
   Do not guess or substitute. Use ONLY: {customer_id}

YOUR TASK — call tools in this exact order with EXACTLY "{customer_id}":
1. lookup_customer("{customer_id}")
2. check_account_status("{customer_id}")
3. list_enabled_features("{customer_id}")
4. get_billing_history("{customer_id}") — only if billing is relevant to the query
5. Note any discrepancies found in the results above.
6. If any tool fails, document it and continue with available data.

OUTPUT FORMAT — always end with a structured findings block:
ACCOUNT_FINDINGS:
- Plan: [plan name]
- Status: [active/suspended/overdue]
- Seats Licensed: [N] | Seats Used: [N] | Overage: [N or none]
- Features Enabled: [comma-separated list]
- Contract ID: [CTR-XXX]
- Billing Status: [current/overdue/unknown]
- Discrepancies Found: [list or 'none']
- Tool Failures: [list or 'none']
- Confidence: [high/medium/low]
""",
        agent=agent,
        expected_output=(
            "Only the ACCOUNT_FINDINGS block. "
            "No 'Thought:' text, no 'Action:' text, no preamble, no code fences. "
            "Start directly with 'ACCOUNT_FINDINGS:' and fill in every field."
        ),
    )
