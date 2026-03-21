"""
account_tools.py
Mock tools for the Account Agent.
Simulates a real customer database with realistic data and edge cases.
"""

import random
import logging
from typing import Optional
from crewai.tools import tool
from tools.tool_base import with_tool_behavior, format_tool_result

logger = logging.getLogger(__name__)

# ── Mock Customer Database ─────────────────────────────────────────────────────

MOCK_CUSTOMERS = {
    "CUST-001": {
        "customer_id": "CUST-001",
        "company_name": "Acme Corp",
        "contact_name": "Jane Doe",
        "email": "jane@acmecorp.com",
        "plan": "starter",
        "seats": 10,
        "active_seats": 15,     # intentional mismatch for scenario 5
        "status": "active",
        "contract_id": "CTR-001",
        "created_at": "2023-01-15",
        "renewal_date": "2025-01-15",
        "mrr": 49.00,
    },
    "CUST-002": {
        "customer_id": "CUST-002",
        "company_name": "Globex Inc",
        "contact_name": "John Smith",
        "email": "john@globex.com",
        "plan": "pro",
        "seats": 50,
        "active_seats": 47,
        "status": "active",
        "contract_id": "CTR-002",
        "created_at": "2022-06-01",
        "renewal_date": "2025-06-01",
        "mrr": 299.00,
    },
    "CUST-003": {
        "customer_id": "CUST-003",
        "company_name": "Initech LLC",
        "contact_name": "Bill Lumbergh",
        "email": "bill@initech.com",
        "plan": "enterprise",
        "seats": 200,
        "active_seats": 180,
        "status": "active",
        "contract_id": "CTR-003",
        "created_at": "2021-03-10",
        "renewal_date": "2026-03-10",
        "mrr": 1499.00,
    },
    "CUST-004": {
        "customer_id": "CUST-004",
        "company_name": "Umbrella Ltd",
        "contact_name": "Alice Wesker",
        "email": "alice@umbrella.com",
        "plan": "pro",
        "seats": 20,
        "active_seats": 18,
        "status": "suspended",
        "contract_id": "CTR-004",
        "created_at": "2023-09-01",
        "renewal_date": "2024-09-01",
        "mrr": 149.00,
    },
}

MOCK_BILLING = {
    "CUST-001": [
        {"date": "2025-01-01", "amount": 49.00, "status": "paid", "invoice": "INV-1001"},
        {"date": "2024-12-01", "amount": 49.00, "status": "paid", "invoice": "INV-1000"},
        {"date": "2024-11-01", "amount": 49.00, "status": "paid", "invoice": "INV-0999"},
    ],
    "CUST-002": [
        {"date": "2025-01-01", "amount": 299.00, "status": "paid", "invoice": "INV-2001"},
        {"date": "2024-12-01", "amount": 299.00, "status": "paid", "invoice": "INV-2000"},
        {"date": "2024-11-01", "amount": 299.00, "status": "paid", "invoice": "INV-1999"},
    ],
    "CUST-003": [
        {"date": "2025-01-01", "amount": 1499.00, "status": "paid", "invoice": "INV-3001"},
        {"date": "2024-12-01", "amount": 1499.00, "status": "paid", "invoice": "INV-3000"},
    ],
    "CUST-004": [
        {"date": "2024-09-01", "amount": 149.00, "status": "overdue", "invoice": "INV-4001"},
        {"date": "2024-08-01", "amount": 149.00, "status": "paid", "invoice": "INV-4000"},
    ],
}

MOCK_ENABLED_FEATURES = {
    "CUST-001": ["basic_dashboard", "csv_export", "email_support"],
    "CUST-002": ["basic_dashboard", "advanced_analytics", "api_access",
                 "csv_export", "priority_support", "webhooks"],
    "CUST-003": ["basic_dashboard", "advanced_analytics", "api_access",
                 "csv_export", "priority_support", "webhooks", "sso",
                 "custom_roles", "dark_mode", "dedicated_support"],
    "CUST-004": ["basic_dashboard", "api_access", "priority_support"],
}


# ── Tool Implementations ───────────────────────────────────────────────────────

@with_tool_behavior("lookup_customer")
def _lookup_customer_impl(customer_id: str) -> dict:
    customer = MOCK_CUSTOMERS.get(customer_id.upper())
    if not customer:
        return {"found": False, "customer_id": customer_id, "message": "Customer not found in database."}
    # Occasionally return incomplete data
    if random.random() < 0.10:
        partial = {k: v for k, v in customer.items() if k not in ["mrr", "renewal_date"]}
        partial["_data_note"] = "Partial record returned — billing fields unavailable."
        return partial
    return customer


@with_tool_behavior("get_billing_history")
def _get_billing_history_impl(customer_id: str) -> dict:
    history = MOCK_BILLING.get(customer_id.upper())
    if not history:
        return {"found": False, "customer_id": customer_id, "billing_records": []}
    return {"customer_id": customer_id, "billing_records": history, "record_count": len(history)}


@with_tool_behavior("check_account_status")
def _check_account_status_impl(customer_id: str) -> dict:
    customer = MOCK_CUSTOMERS.get(customer_id.upper())
    if not customer:
        return {"found": False, "customer_id": customer_id}
    return {
        "customer_id": customer_id,
        "status": customer["status"],
        "plan": customer["plan"],
        "seats_licensed": customer["seats"],
        "seats_active": customer["active_seats"],
        "seats_available": max(0, customer["seats"] - customer["active_seats"]),
        "seat_overage": max(0, customer["active_seats"] - customer["seats"]),
        "renewal_date": customer.get("renewal_date", "unknown"),
    }


@with_tool_behavior("list_enabled_features")
def _list_enabled_features_impl(customer_id: str) -> dict:
    features = MOCK_ENABLED_FEATURES.get(customer_id.upper())
    if features is None:
        return {"found": False, "customer_id": customer_id, "features": []}
    return {
        "customer_id": customer_id,
        "enabled_features": features,
        "feature_count": len(features),
    }


# ── CrewAI Tool Wrappers ───────────────────────────────────────────────────────

@tool("Lookup Customer Profile")
def lookup_customer(customer_id: str) -> str:
    """
    Fetch complete customer profile from the database.
    Input: customer_id (string, e.g. 'CUST-001')
    Returns: customer profile including plan, seats, status, contract ID.
    Use when you need to identify the customer's current plan and account details.
    """
    result = _lookup_customer_impl(customer_id)
    return format_tool_result(result)


@tool("Get Billing History")
def get_billing_history(customer_id: str) -> str:
    """
    Retrieve billing records for a customer.
    Input: customer_id (string)
    Returns: list of invoices with dates, amounts, and payment status.
    Use when investigating billing discrepancies or payment history.
    """
    result = _get_billing_history_impl(customer_id)
    return format_tool_result(result)


@tool("Check Account Status")
def check_account_status(customer_id: str) -> str:
    """
    Verify the current account status, seat allocation, and renewal date.
    Input: customer_id (string)
    Returns: status (active/suspended), seat counts, renewal date.
    Use when checking if an account is in good standing or has seat overages.
    """
    result = _check_account_status_impl(customer_id)
    return format_tool_result(result)


@tool("List Enabled Features")
def list_enabled_features(customer_id: str) -> str:
    """
    Get the list of features currently enabled on a customer's account.
    Input: customer_id (string)
    Returns: list of enabled feature names.
    Use when verifying what features a customer actually has access to.
    """
    result = _list_enabled_features_impl(customer_id)
    return format_tool_result(result)
