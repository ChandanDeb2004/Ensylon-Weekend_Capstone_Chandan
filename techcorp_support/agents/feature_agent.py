"""
feature_agent.py
Feature Agent — checks feature availability, documentation, limits, and configurations.
"""

import logging
from crewai import Agent, Task
from agents.base_agent import build_agent
from tools.feature_tools import (
    get_feature_matrix, get_feature_documentation,
    validate_configuration, check_feature_limits,
)
from tools.account_tools import lookup_customer, list_enabled_features

logger = logging.getLogger(__name__)

FEATURE_AGENT_ROLE = "Feature Availability and Configuration Specialist"

FEATURE_AGENT_GOAL = """
Answer feature questions with precision using real data from tools.
Your job is to:
- Look up the customer's actual plan before checking feature availability
- Retrieve feature documentation and setup steps
- Check rate limits and entitlements per plan
- Surface documentation discrepancies (especially DOC-1182: Pro API limit is 1,000/month not unlimited)
- Give concrete upgrade recommendations when features are unavailable

Never say "unknown" for Available on Customer Plan if you have a customer_id —
always call lookup_customer first to get the real plan.
"""

FEATURE_AGENT_BACKSTORY = """
You are TechCorp's lead product specialist with encyclopedic knowledge of every feature
across all subscription tiers. You maintain the feature matrix and are the first person
engineers consult when there's a discrepancy between documentation and actual system behavior.

You are acutely aware that TechCorp's public documentation has a known error: it states
Pro plan has 'unlimited API calls' but the actual enforced limit is 1,000/month (tracked
as DOC-1182). You proactively surface this when relevant.

When a customer cannot access a feature, you methodically check:
(1) What is their actual plan? (call lookup_customer if not known)
(2) Is the feature on that plan? (call get_feature_matrix)
(3) Are there rate limits? (call check_feature_limits)
(4) Is it configured correctly? (call validate_configuration if needed)

You give clear, specific answers. If the plan is Enterprise, dark mode IS available.
If the plan is Starter, API is NOT available. Never hedge if you have real data.
"""


def create_feature_agent(session_id: str = "default") -> Agent:
    return build_agent(
        role=FEATURE_AGENT_ROLE,
        goal=FEATURE_AGENT_GOAL,
        backstory=FEATURE_AGENT_BACKSTORY,
        tools=[
            lookup_customer,
            list_enabled_features,
            get_feature_matrix,
            get_feature_documentation,
            validate_configuration,
            check_feature_limits,
        ],
        session_id=session_id,
        max_iter=10,
    )


def create_feature_task(
    agent: Agent,
    query: str,
    context_str: str,
    feature_name: str = "",
    customer_plan: str = "",
    customer_id: str = "",
) -> Task:
    feature_hint = f"The query is specifically about: '{feature_name}'" if feature_name else ""
    plan_hint    = f"Customer plan already known: '{customer_plan}' — skip lookup_customer." if customer_plan else ""
    cid_hint     = f"Customer ID: {customer_id}" if customer_id else ""

    # If plan unknown and customer_id available, instruct to look it up
    if not customer_plan and customer_id:
        lookup_instruction = f"Step 1: Call lookup_customer(\"{customer_id}\") to get their plan — you MUST do this before checking feature availability."
    elif customer_plan:
        lookup_instruction = f"Step 1: Customer is on {customer_plan} plan (already known — skip lookup_customer)."
    else:
        lookup_instruction = "Step 1: Use the CUSTOMER_ID from context to call lookup_customer and get their plan."

    return Task(
        description=f"""
Answer this feature support query using real data from tools.

QUERY: {query}

SHARED CONTEXT:
{context_str}

{cid_hint}
{feature_hint}
{plan_hint}

EXECUTION STEPS:
{lookup_instruction}
Step 2: Call get_feature_matrix() to see which plans include which features.
Step 3: If a specific feature is mentioned, call get_feature_documentation(feature_name).
Step 4: Call check_feature_limits(feature, plan) using the customer's REAL plan from Step 1.
Step 5: Call validate_configuration(feature, config) only if the customer reports a config problem.

RULES:
- "Available on Customer Plan" MUST use the real plan from Step 1 — never write "unknown" if you have a customer_id
- Fill every field with real tool data, not placeholders
- Flag DOC-1182 (Pro API = 1,000/month not unlimited) when relevant
- If a tool fails, write the actual error and continue with available data

OUTPUT — start directly with this block, no preamble:

FEATURE_FINDINGS:
- Feature Investigated: [exact feature name from query]
- Available on Customer Plan: [yes/no — based on real plan from lookup_customer]
- Customer Plan: [actual plan name: starter/pro/enterprise]
- Plan Required for Feature: [plan name(s) that include this feature]
- Rate Limit: [specific limit or 'none']
- Setup Steps: [numbered steps or 'N/A']
- Documentation Discrepancies: [describe or 'none found']
- Configuration Issues: [describe or 'none']
- Upgrade Recommendation: [specific upgrade path or 'not needed']
- Tool Failures: [list or 'none']
- Confidence: [high/medium/low]
""",
        agent=agent,
        expected_output=(
            "Only the FEATURE_FINDINGS block with all fields filled using real tool data. "
            "No 'Thought:' text, no 'Action:' text, no preamble, no code fences. "
            "'Available on Customer Plan' must say yes or no, never unknown. "
            "Start directly with 'FEATURE_FINDINGS:' on the first line."
        ),
    )
