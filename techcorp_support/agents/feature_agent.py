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

logger = logging.getLogger(__name__)

FEATURE_AGENT_ROLE = "Feature Availability and Configuration Specialist"

FEATURE_AGENT_GOAL = """
Investigate feature availability, plan entitlements, configuration correctness, and known
documentation issues. Provide precise answers about:
- Whether a feature is available on the customer's current plan
- Step-by-step setup instructions for features
- Rate limits and usage caps (including known documentation discrepancies)
- Configuration validation for features the customer is struggling with
- Upgrade path recommendations when a feature is not on their current plan

Flag any known documentation discrepancies explicitly — do not let incorrect docs mislead
the investigation.
"""

FEATURE_AGENT_BACKSTORY = """
You are TechCorp's lead product specialist with encyclopedic knowledge of every feature
across all subscription tiers. You maintain the feature matrix and are the first person
engineers consult when there's a discrepancy between documentation and actual system behavior.

You are acutely aware that TechCorp's public documentation has a known error: it states
Pro plan has 'unlimited API calls' but the actual enforced limit is 1,000/month (tracked
as DOC-1182). You proactively surface this when relevant.

When a customer cannot access a feature, you methodically check: (1) is it on their plan?
(2) is it configured correctly? (3) are there rate limits being hit? You give clear,
actionable guidance — never vague answers.
"""


def create_feature_agent(session_id: str = "default") -> Agent:
    return build_agent(
        role=FEATURE_AGENT_ROLE,
        goal=FEATURE_AGENT_GOAL,
        backstory=FEATURE_AGENT_BACKSTORY,
        tools=[
            get_feature_matrix,
            get_feature_documentation,
            validate_configuration,
            check_feature_limits,
        ],
        session_id=session_id,
        max_iter=8,
    )


def create_feature_task(
    agent: Agent,
    query: str,
    context_str: str,
    feature_name: str = "",
    customer_plan: str = "",
) -> Task:
    feature_hint = f"Focus especially on feature: '{feature_name}'" if feature_name else ""
    plan_hint = f"Customer is on plan: '{customer_plan}'" if customer_plan else ""

    return Task(
        description=f"""
Investigate feature availability and configuration for this support query:

QUERY: {query}

SHARED INVESTIGATION CONTEXT:
{context_str}

{feature_hint}
{plan_hint}

YOUR TASK:
1. Call get_feature_matrix() to understand what features exist on which plans.
2. If a specific feature is mentioned, call get_feature_documentation(feature_name) for
   detailed docs, setup steps, and known issues.
3. If the customer's plan is known, call check_feature_limits(feature, plan) to get
   rate limits and entitlement details.
4. If the customer has a configuration problem, call validate_configuration(feature, config).
5. Explicitly flag any DOCUMENTATION DISCREPANCIES you find (especially API call limits).
6. If a feature is unavailable on their plan, clearly state what upgrade would enable it.
7. If any tool fails, note it and proceed with available information.

OUTPUT FORMAT — always end with a structured findings block:
FEATURE_FINDINGS:
- Feature Investigated: [feature name]
- Available on Customer Plan: [yes/no/unknown]
- Plan Required: [plan name(s)]
- Rate Limit: [limit or 'none' or 'unknown']
- Setup Steps: [brief steps or 'N/A']
- Documentation Discrepancies: [describe or 'none found']
- Configuration Issues: [describe or 'none']
- Upgrade Recommendation: [describe or 'not needed']
- Tool Failures: [list or 'none']
- Confidence: [high/medium/low]
""",
        agent=agent,
        expected_output=(
            "A structured feature investigation report with availability, limits, "
            "setup guidance, any documentation discrepancies, and upgrade recommendations."
        ),
    )
