"""
feature_tools.py
Mock tools for the Feature Agent.
Simulates a feature matrix database with plan-based entitlements.
"""

import logging
from crewai.tools import tool
from tools.tool_base import with_tool_behavior, format_tool_result

logger = logging.getLogger(__name__)

# ── Mock Feature Data ──────────────────────────────────────────────────────────

FEATURE_MATRIX = {
    "basic_dashboard": {
        "plans": ["starter", "pro", "enterprise"],
        "description": "Core analytics dashboard with standard views.",
    },
    "dark_mode": {
        "plans": ["enterprise"],
        "description": "Dark UI theme. Enable under Settings → Appearance → Theme → Dark.",
        "setup_steps": [
            "1. Log in as an Admin.",
            "2. Go to Settings → Appearance.",
            "3. Under Theme, select 'Dark'.",
            "4. Click Save. Changes apply immediately for all users.",
        ],
    },
    "api_access": {
        "plans": ["pro", "enterprise"],
        "description": "REST API access for programmatic integration.",
        "limits": {
            "starter": None,
            "pro": "1000 calls/month (note: documentation may say unlimited — this is a known discrepancy)",
            "enterprise": "unlimited",
        },
    },
    "csv_export": {
        "plans": ["starter", "pro", "enterprise"],
        "description": "Export data as CSV files.",
    },
    "advanced_analytics": {
        "plans": ["pro", "enterprise"],
        "description": "Advanced reporting, custom charts, and trend analysis.",
    },
    "webhooks": {
        "plans": ["pro", "enterprise"],
        "description": "Real-time event webhooks for external integrations.",
    },
    "sso": {
        "plans": ["enterprise"],
        "description": "Single Sign-On via SAML 2.0 or OAuth 2.0.",
    },
    "custom_roles": {
        "plans": ["enterprise"],
        "description": "Define custom user roles and permission sets.",
    },
    "priority_support": {
        "plans": ["pro", "enterprise"],
        "description": "Priority access to support queue, 4-hour SLA.",
    },
    "dedicated_support": {
        "plans": ["enterprise"],
        "description": "Dedicated Customer Success Manager and 24-hour SLA.",
    },
    "email_support": {
        "plans": ["starter", "pro", "enterprise"],
        "description": "Email-based support with 2-business-day response.",
    },
}

FEATURE_DOCS = {
    "api_access": """
API Access Documentation
========================
The TechCorp REST API allows you to programmatically access your data.

IMPORTANT NOTE: Our public documentation states 'unlimited API calls' for Pro plan.
This is INCORRECT. The current enforced limit is 1,000 calls/month for Pro.
Enterprise plans have true unlimited access. This documentation discrepancy is a
known issue tracked in ticket DOC-1182.

Authentication: Bearer token (generate under Settings → API → Tokens).
Base URL: https://api.techcorp.io/v1/
Rate Limiting: 429 status returned when limit is exceeded.
""",
    "dark_mode": """
Dark Mode Setup Guide
=====================
Dark Mode is available exclusively on Enterprise plans.

To enable:
1. Log in with Admin credentials.
2. Navigate to Settings → Appearance.
3. Select Theme: Dark.
4. Save changes. Theme applies organization-wide immediately.

Troubleshooting: If the option is greyed out, verify your plan includes this feature.
""",
    "sso": """
SSO Configuration Guide
========================
Supports SAML 2.0 and OAuth 2.0. Available on Enterprise only.
Contact your dedicated support manager to initiate SSO setup.
""",
}

PLAN_SEAT_LIMITS = {
    "starter": 10,
    "pro": 100,
    "enterprise": "unlimited",
}

PLAN_PRICING = {
    "starter": {"price": 49, "billing": "monthly"},
    "pro": {"price": 299, "billing": "monthly"},
    "enterprise": {"price": "custom", "billing": "annual contract"},
}


# ── Tool Implementations ───────────────────────────────────────────────────────

@with_tool_behavior("get_feature_matrix")
def _get_feature_matrix_impl() -> dict:
    matrix = {}
    for feature, data in FEATURE_MATRIX.items():
        matrix[feature] = {
            "available_on_plans": data["plans"],
            "description": data["description"],
        }
    return {
        "feature_matrix": matrix,
        "plans": ["starter", "pro", "enterprise"],
        "plan_pricing": PLAN_PRICING,
        "seat_limits": PLAN_SEAT_LIMITS,
    }


@with_tool_behavior("get_feature_documentation")
def _get_feature_documentation_impl(feature_name: str) -> dict:
    feature = FEATURE_MATRIX.get(feature_name.lower().replace(" ", "_"))
    doc = FEATURE_DOCS.get(feature_name.lower().replace(" ", "_"))
    if not feature:
        return {"found": False, "feature": feature_name, "message": "Feature not found."}
    return {
        "feature": feature_name,
        "description": feature.get("description"),
        "available_on": feature.get("plans"),
        "setup_steps": feature.get("setup_steps", []),
        "documentation": doc or "No extended documentation available for this feature.",
        "limits": feature.get("limits", {}),
    }


@with_tool_behavior("validate_configuration")
def _validate_configuration_impl(feature: str, customer_config: str) -> dict:
    feature_key = feature.lower().replace(" ", "_")
    known = FEATURE_MATRIX.get(feature_key)
    if not known:
        return {"valid": False, "reason": f"Unknown feature: {feature}"}
    # Simulate config validation logic
    config_lower = customer_config.lower()
    issues = []
    if "api_access" in feature_key and "token" not in config_lower:
        issues.append("No API token found in configuration.")
    if "sso" in feature_key and "saml" not in config_lower and "oauth" not in config_lower:
        issues.append("SSO requires SAML or OAuth configuration.")
    return {
        "feature": feature,
        "config_received": customer_config,
        "is_valid": len(issues) == 0,
        "issues_found": issues,
        "recommendation": "Fix the issues above before testing." if issues else "Configuration looks correct.",
    }


@with_tool_behavior("check_feature_limits")
def _check_feature_limits_impl(feature: str, plan: str) -> dict:
    feature_key = feature.lower().replace(" ", "_")
    known = FEATURE_MATRIX.get(feature_key)
    if not known:
        return {"found": False, "feature": feature, "plan": plan}
    plan_lower = plan.lower()
    limits = known.get("limits", {})
    available = plan_lower in known.get("plans", [])
    return {
        "feature": feature,
        "plan": plan,
        "available": available,
        "limit": limits.get(plan_lower, "no specific limit defined"),
        "available_on_plans": known.get("plans", []),
        "upgrade_required": not available,
    }


# ── CrewAI Tool Wrappers ───────────────────────────────────────────────────────

@tool("Get Feature Matrix")
def get_feature_matrix(dummy: str = "") -> str:
    """
    Retrieve the complete feature availability matrix across all plans.
    Input: none required (dummy parameter exists for API compatibility only, leave blank).
    Returns: feature list, which plans include each feature, pricing, seat limits.
    Use when comparing what features are available on different subscription tiers.
    """
    result = _get_feature_matrix_impl()
    return format_tool_result(result)


@tool("Get Feature Documentation")
def get_feature_documentation(feature_name: str) -> str:
    """
    Fetch detailed documentation for a specific feature.
    Input: feature_name (string, e.g. 'dark_mode', 'api_access', 'sso')
    Returns: description, setup steps, plan availability, known issues.
    Use when a customer needs setup instructions or detailed feature information.
    """
    result = _get_feature_documentation_impl(feature_name)
    return format_tool_result(result)


@tool("Validate Feature Configuration")
def validate_configuration(feature: str, customer_config: str) -> str:
    """
    Validate whether a customer's configuration for a feature is correct.
    Input: feature (string), customer_config (string describing their setup)
    Returns: validation result, issues found, recommendations.
    Use when a customer reports a feature isn't working as expected.
    """
    result = _validate_configuration_impl(feature, customer_config)
    return format_tool_result(result)


@tool("Check Feature Limits")
def check_feature_limits(feature: str, plan: str) -> str:
    """
    Get rate limits and caps for a specific feature on a given plan.
    Input: feature (string), plan (string: 'starter', 'pro', or 'enterprise')
    Returns: limit details, whether feature is available, upgrade info.
    Use when investigating rate limit errors or plan entitlement questions.
    """
    result = _check_feature_limits_impl(feature, plan)
    return format_tool_result(result)
