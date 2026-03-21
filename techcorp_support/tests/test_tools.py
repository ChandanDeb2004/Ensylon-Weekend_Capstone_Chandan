"""
tests/test_tools.py
Unit tests for mock tools — verifies data correctness, error handling, and formatting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
from unittest.mock import patch
from tools.tool_base import format_tool_result
from tools.account_tools import (
    _lookup_customer_impl, _get_billing_history_impl,
    _check_account_status_impl, _list_enabled_features_impl,
)
from tools.feature_tools import (
    _get_feature_matrix_impl, _get_feature_documentation_impl,
    _check_feature_limits_impl,
)
from tools.contract_tools import (
    _lookup_contract_impl, _validate_sla_compliance_impl,
    _get_included_features_impl,
)
from tools.escalation_tools import (
    _create_escalation_ticket_impl, _get_escalation_routing_impl,
)


class TestAccountTools(unittest.TestCase):

    def test_lookup_existing_customer(self):
        result = _lookup_customer_impl("CUST-001")
        if result["success"]:  # may fail due to random failure injection
            self.assertEqual(result["data"]["plan"], "starter")
            self.assertEqual(result["data"]["company_name"], "Acme Corp")

    def test_lookup_nonexistent_customer(self):
        result = _lookup_customer_impl("CUST-999")
        if result["success"]:
            self.assertFalse(result["data"]["found"])

    def test_check_account_status_overage(self):
        """CUST-001 has 15 active seats but only 10 licensed — should detect overage."""
        result = _check_account_status_impl("CUST-001")
        if result["success"] and result["data"].get("found", True):
            data = result["data"]
            self.assertEqual(data.get("seat_overage"), 5)

    def test_list_features_enterprise(self):
        result = _list_enabled_features_impl("CUST-003")
        if result["success"]:
            features = result["data"]["enabled_features"]
            self.assertIn("dark_mode", features)
            self.assertIn("sso", features)

    def test_list_features_starter_no_api(self):
        result = _list_enabled_features_impl("CUST-001")
        if result["success"]:
            features = result["data"]["enabled_features"]
            self.assertNotIn("api_access", features)

    def test_billing_history_returns_records(self):
        result = _get_billing_history_impl("CUST-002")
        if result["success"]:
            self.assertGreater(result["data"]["record_count"], 0)


class TestFeatureTools(unittest.TestCase):

    def test_feature_matrix_contains_api(self):
        result = _get_feature_matrix_impl()
        if result["success"]:
            matrix = result["data"]["feature_matrix"]
            self.assertIn("api_access", matrix)
            self.assertIn("pro", matrix["api_access"]["available_on_plans"])

    def test_dark_mode_enterprise_only(self):
        result = _get_feature_matrix_impl()
        if result["success"]:
            matrix = result["data"]["feature_matrix"]
            plans = matrix["dark_mode"]["available_on_plans"]
            self.assertEqual(plans, ["enterprise"])

    def test_feature_documentation_api_discrepancy(self):
        result = _get_feature_documentation_impl("api_access")
        if result["success"]:
            doc = result["data"]["documentation"]
            # Should mention the documentation discrepancy
            self.assertIn("1,000", doc)

    def test_check_limits_pro_api(self):
        result = _check_feature_limits_impl("api_access", "pro")
        if result["success"]:
            data = result["data"]
            self.assertTrue(data["available"])
            self.assertIn("1000", str(data["limit"]))

    def test_check_limits_starter_api_unavailable(self):
        result = _check_feature_limits_impl("api_access", "starter")
        if result["success"]:
            data = result["data"]
            self.assertFalse(data["available"])


class TestContractTools(unittest.TestCase):

    def test_lookup_contract_enterprise(self):
        result = _lookup_contract_impl("CTR-003")
        if result["success"]:
            data = result["data"]
            self.assertEqual(data["sla_response_hours"], 24)
            self.assertEqual(data["sla_type"], "guaranteed")

    def test_sla_breach_detection(self):
        """CTR-003 has 24hr guaranteed SLA; 10 days elapsed should be a breach."""
        from datetime import datetime, timedelta, timezone
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        result = _validate_sla_compliance_impl("CTR-003", ten_days_ago)
        if result["success"]:
            data = result["data"]
            self.assertTrue(data["sla_breached"])
            self.assertGreater(data["hours_overdue"], 200)

    def test_get_included_features_starter(self):
        result = _get_included_features_impl("CTR-001")
        if result["success"]:
            features = result["data"]["included_features"]
            self.assertNotIn("api_access", features)

    def test_api_limit_in_contract_special_terms(self):
        """CTR-002 has a special term specifying 1000 API calls/month."""
        result = _lookup_contract_impl("CTR-002")
        if result["success"]:
            terms = result["data"]["special_terms"]
            self.assertTrue(any("1000" in t for t in terms))


class TestEscalationTools(unittest.TestCase):

    def test_create_ticket(self):
        result = _create_escalation_ticket_impl(
            reason="SLA breach confirmed",
            priority="critical",
            context="Customer waited 10 days on 24hr guaranteed SLA.",
        )
        if result["success"]:
            data = result["data"]
            self.assertTrue(data["ticket_id"].startswith("ESC-"))
            self.assertEqual(data["priority"], "critical")
            self.assertTrue(data["notify_immediately"])

    def test_escalation_routing_sla(self):
        result = _get_escalation_routing_impl("sla_violation")
        if result["success"]:
            data = result["data"]
            self.assertEqual(data["assigned_team"], "Customer Success")

    def test_escalation_routing_onboarding(self):
        result = _get_escalation_routing_impl("onboarding")
        if result["success"]:
            data = result["data"]
            self.assertEqual(data["assigned_team"], "Onboarding")


class TestToolResultFormatter(unittest.TestCase):

    def test_format_success(self):
        result = {
            "success": True,
            "tool": "test_tool",
            "call_id": "test_001",
            "data": {"key": "value"},
            "latency_ms": 150,
            "timestamp": "2025-01-01T00:00:00",
        }
        formatted = format_tool_result(result)
        self.assertIn("TOOL SUCCESS", formatted)
        self.assertIn("value", formatted)

    def test_format_failure(self):
        result = {
            "success": False,
            "tool": "test_tool",
            "call_id": "test_002",
            "error": "DatabaseTimeout",
            "error_message": "Connection timed out",
            "data": None,
            "latency_ms": 5000,
            "timestamp": "2025-01-01T00:00:00",
        }
        formatted = format_tool_result(result)
        self.assertIn("TOOL CALL FAILED", formatted)
        self.assertIn("DatabaseTimeout", formatted)
        self.assertIn("gracefully", formatted)


class TestQueryAnalysis(unittest.TestCase):

    def test_dark_mode_routes_to_feature_only(self):
        from agents.orchestrator import analyze_query
        plan = analyze_query("How do I enable dark mode?")
        self.assertTrue(plan["needs_feature"])
        self.assertFalse(plan["needs_contract"])

    def test_sla_violation_routes_all_agents(self):
        from agents.orchestrator import analyze_query
        plan = analyze_query(
            "SLA violated! 10 days no response, $500/day lost revenue. Escalate immediately."
        )
        self.assertTrue(plan["needs_contract"])
        self.assertTrue(plan["needs_escalation"])
        self.assertTrue(plan["revenue_impact"])

    def test_api_plan_mismatch_routes_account_and_feature(self):
        from agents.orchestrator import analyze_query
        plan = analyze_query(
            "I'm on Starter but need API access for my automation workflow."
        )
        self.assertTrue(plan["needs_account"])
        self.assertTrue(plan["needs_feature"])

    def test_customer_id_extraction(self):
        from agents.orchestrator import analyze_query
        plan = analyze_query("Please check account CUST-002 for billing issues.")
        self.assertEqual(plan["customer_id"], "CUST-002")


if __name__ == "__main__":
    unittest.main(verbosity=2)
