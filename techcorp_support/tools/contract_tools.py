"""
contract_tools.py
Mock tools for the Contract Agent.
Simulates a contract management database with SLA validation.
"""

import logging
from datetime import datetime, timedelta, timezone
from crewai.tools import tool
from tools.tool_base import with_tool_behavior, format_tool_result

logger = logging.getLogger(__name__)

# ── Mock Contract Database ─────────────────────────────────────────────────────

MOCK_CONTRACTS = {
    "CTR-001": {
        "contract_id": "CTR-001",
        "customer_id": "CUST-001",
        "company": "Acme Corp",
        "plan": "starter",
        "start_date": "2023-01-15",
        "end_date": "2025-01-15",
        "sla_response_hours": 48,
        "sla_type": "best_effort",
        "included_features": ["basic_dashboard", "csv_export", "email_support"],
        "api_calls_per_month": 0,
        "seats": 10,
        "monthly_price": 49.00,
        "special_terms": [],
    },
    "CTR-002": {
        "contract_id": "CTR-002",
        "customer_id": "CUST-002",
        "company": "Globex Inc",
        "plan": "pro",
        "start_date": "2022-06-01",
        "end_date": "2025-06-01",
        "sla_response_hours": 4,
        "sla_type": "guaranteed",
        "included_features": [
            "basic_dashboard", "advanced_analytics", "api_access",
            "csv_export", "priority_support", "webhooks",
        ],
        "api_calls_per_month": 1000,  # actual limit — docs say unlimited (known conflict)
        "seats": 50,
        "monthly_price": 299.00,
        "special_terms": ["API limit: 1000 calls/month as per pricing addendum dated 2022-06-01"],
    },
    "CTR-003": {
        "contract_id": "CTR-003",
        "customer_id": "CUST-003",
        "company": "Initech LLC",
        "plan": "enterprise",
        "start_date": "2021-03-10",
        "end_date": "2026-03-10",
        "sla_response_hours": 24,
        "sla_type": "guaranteed",
        "included_features": [
            "basic_dashboard", "advanced_analytics", "api_access",
            "csv_export", "priority_support", "webhooks", "sso",
            "custom_roles", "dark_mode", "dedicated_support",
        ],
        "api_calls_per_month": "unlimited",
        "seats": 200,
        "monthly_price": 1499.00,
        "special_terms": [
            "24-hour guaranteed SLA for critical issues",
            "Dedicated CSM assigned: support-manager@techcorp.io",
            "SLA breach compensation: 10% service credit per breach",
        ],
    },
    "CTR-004": {
        "contract_id": "CTR-004",
        "customer_id": "CUST-004",
        "company": "Umbrella Ltd",
        "plan": "pro",
        "start_date": "2023-09-01",
        "end_date": "2024-09-01",
        "sla_response_hours": 4,
        "sla_type": "guaranteed",
        "included_features": ["basic_dashboard", "api_access", "priority_support"],
        "api_calls_per_month": 1000,
        "seats": 20,
        "monthly_price": 149.00,
        "special_terms": [],
    },
}

SLA_RECORDS = {
    "CTR-003": {
        "open_issues": [
            {
                "ticket_id": "TKT-9981",
                "issue": "Critical production outage",
                "opened_at": (datetime.now(timezone.utc)- timedelta(days=10)).isoformat(),
                "responded_at": None,
                "resolved_at": None,
                "severity": "critical",
            }
        ]
    }
}


# ── Tool Implementations ───────────────────────────────────────────────────────

@with_tool_behavior("lookup_contract")
def _lookup_contract_impl(contract_id: str) -> dict:
    contract = MOCK_CONTRACTS.get(contract_id.upper())
    if not contract:
        return {"found": False, "contract_id": contract_id, "message": "Contract not found."}
    return contract


@with_tool_behavior("get_contract_terms")
def _get_contract_terms_impl(contract_id: str) -> dict:
    contract = MOCK_CONTRACTS.get(contract_id.upper())
    if not contract:
        return {"found": False, "contract_id": contract_id}
    return {
        "contract_id": contract_id,
        "sla_response_hours": contract["sla_response_hours"],
        "sla_type": contract["sla_type"],
        "seats": contract["seats"],
        "monthly_price": contract["monthly_price"],
        "api_calls_per_month": contract["api_calls_per_month"],
        "special_terms": contract["special_terms"],
        "contract_end": contract["end_date"],
    }


@with_tool_behavior("validate_sla_compliance")
def _validate_sla_compliance_impl(contract_id: str, issue_date: str) -> dict:
    contract = MOCK_CONTRACTS.get(contract_id.upper())
    if not contract:
        return {"found": False, "contract_id": contract_id}

    try:
        issue_dt = datetime.fromisoformat(issue_date)
    except ValueError:
        issue_dt = datetime.now(timezone.utc) - timedelta(days=10)

    now = datetime.now(timezone.utc)
    sla_hours = contract["sla_response_hours"]
    sla_deadline = issue_dt + timedelta(hours=sla_hours)
    hours_elapsed = (now - issue_dt).total_seconds() / 3600
    sla_breached = now > sla_deadline
    hours_overdue = max(0, (now - sla_deadline).total_seconds() / 3600)

    # Check if there's an open SLA record
    sla_record = SLA_RECORDS.get(contract_id.upper(), {})
    open_issues = sla_record.get("open_issues", [])

    return {
        "contract_id": contract_id,
        "sla_type": contract["sla_type"],
        "sla_response_hours": sla_hours,
        "issue_opened": issue_date,
        "sla_deadline": sla_deadline.isoformat(),
        "hours_elapsed": round(hours_elapsed, 1),
        "sla_breached": sla_breached,
        "hours_overdue": round(hours_overdue, 1) if sla_breached else 0,
        "open_unresolved_issues": open_issues,
        "compensation_clause": contract.get("special_terms", []),
        "severity": "critical" if sla_breached and contract["sla_type"] == "guaranteed" else "normal",
    }


@with_tool_behavior("get_included_features")
def _get_included_features_impl(contract_id: str) -> dict:
    contract = MOCK_CONTRACTS.get(contract_id.upper())
    if not contract:
        return {"found": False, "contract_id": contract_id}
    return {
        "contract_id": contract_id,
        "plan": contract["plan"],
        "included_features": contract["included_features"],
        "api_calls_per_month": contract["api_calls_per_month"],
        "seats": contract["seats"],
        "special_entitlements": contract["special_terms"],
    }


# ── CrewAI Tool Wrappers ───────────────────────────────────────────────────────

@tool("Lookup Contract")
def lookup_contract(contract_id: str) -> str:
    """
    Fetch full contract details for a customer.
    Input: contract_id (string, e.g. 'CTR-001')
    Returns: plan, SLA terms, pricing, features, special terms.
    Use when verifying what a customer's contract actually specifies.
    """
    result = _lookup_contract_impl(contract_id)
    return format_tool_result(result)


@tool("Get Contract Terms")
def get_contract_terms(contract_id: str) -> str:
    """
    Retrieve specific SLA and pricing terms from a contract.
    Input: contract_id (string)
    Returns: SLA hours, SLA type, pricing, seat count, special terms.
    Use when you need SLA response time commitments or pricing specifics.
    """
    result = _get_contract_terms_impl(contract_id)
    return format_tool_result(result)


@tool("Validate SLA Compliance")
def validate_sla_compliance(contract_id: str, issue_date: str) -> str:
    """
    Check whether a support SLA has been violated for a given issue.
    Input: contract_id (string), issue_date (ISO format string, e.g. '2025-01-01T10:00:00')
    Returns: whether SLA is breached, hours overdue, compensation clauses.
    Use when a customer claims their SLA was violated.
    """
    result = _validate_sla_compliance_impl(contract_id, issue_date)
    return format_tool_result(result)


@tool("Get Included Features")
def get_included_features(contract_id: str) -> str:
    """
    List all features and entitlements included in a specific contract.
    Input: contract_id (string)
    Returns: feature list, API limits, seat count, special entitlements.
    Use when comparing contract entitlements against actual account configuration.
    """
    result = _get_included_features_impl(contract_id)
    return format_tool_result(result)
