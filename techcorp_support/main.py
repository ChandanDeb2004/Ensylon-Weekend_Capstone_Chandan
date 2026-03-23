"""
main.py
CLI entry point for TechCorp Multi-Agent Customer Support System.

Usage:
  python main.py --query "How do I enable dark mode?"
  python main.py --scenario 1
  python main.py --scenario all
  python main.py --list-scenarios
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Langfuse OTEL setup (must run before any agent)
from monitoring.langfuse_config import configure
configure()

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/system.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Ensure results dir exists ──────────────────────────────────────────────────
Path("results").mkdir(exist_ok=True)
Path("results/sessions").mkdir(exist_ok=True)

# ── Test Scenarios ─────────────────────────────────────────────────────────────
TEST_SCENARIOS = {
    1: {
        "name": "Basic Feature Question",
        "query": "How do I enable dark mode in my account?",
        "customer_id": "CUST-003",
        "expected_agents": ["feature", "escalation"],
        "expected_escalation": False,
    },
    2: {
        "name": "Plan-to-Feature Mismatch",
        "query": (
            "I'm on the Starter plan, but I need to integrate with your API for my "
            "automation workflow. What are my options?"
        ),
        "customer_id": "CUST-001",
        "expected_agents": ["account", "feature", "escalation"],
        "expected_escalation": False,
    },
    3: {
        "name": "Contradictory Information",
        "query": (
            "Your documentation says the Pro plan includes unlimited API calls, but I'm "
            "seeing rate limit errors after 1000 calls/month. I've checked my account and "
            "it shows Pro. Is this a bug, or am I misunderstanding something?"
        ),
        "customer_id": "CUST-002",
        "expected_agents": ["account", "feature", "contract", "escalation"],
        "expected_escalation": False,
    },
    4: {
        "name": "SLA Violation",
        "query": (
            "I've been waiting for support response for 10 days on a critical production "
            "issue. My company has a contract with a 24-hour SLA guarantee. This is now "
            "costing us $500/day in lost revenue. I have my contract terms saved. Please "
            "verify if the SLA was violated and escalate this immediately."
        ),
        "customer_id": "CUST-003",
        "expected_agents": ["account", "contract", "escalation"],
        "expected_escalation": True,
    },
    5: {
        "name": "Account Configuration Help",
        "query": (
            "Our company just migrated from the competitor platform. We have 15 users, "
            "but the plan shows only 10 seats. Can you help me understand the licensing "
            "model and figure out how to set up all our users?"
        ),
        "customer_id": "CUST-001",
        "expected_agents": ["account", "feature", "escalation"],
        "expected_escalation": False,
    },
}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║    TechCorp Multi-Agent Customer Support System          ║
║    Backend: CrewAI  |  LLM: qwen2.5 (Ollama)            ║
║    Observability: Langfuse Cloud                         ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_query(query: str, customer_id: str = None, session_id: str = None) -> dict:
    """Import and run the orchestrator."""
    from agents.orchestrator import run_support_crew

    steps = []

    def progress_callback(step: str):
        steps.append(step)
        print(f"  → {step}")

    result = run_support_crew(
        query=query,
        customer_id=customer_id,
        session_id=session_id,
        progress_callback=progress_callback,
    )
    result["progress_steps"] = steps
    return result


def save_results(all_results: list[dict]):
    """Save all scenario results to results/query_results.json."""
    output_path = Path("results/query_results.json")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": len(all_results),
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"[MAIN] Results saved to {output_path}")
    print(f"\n✅ Results saved to {output_path}")


def run_scenario(scenario_num: int) -> dict:
    scenario = TEST_SCENARIOS[scenario_num]
    print(f"\n{'═'*60}")
    print(f"  SCENARIO {scenario_num}: {scenario['name']}")
    print(f"{'═'*60}")
    print(f"  Query: {scenario['query'][:80]}...")
    print(f"  Customer: {scenario['customer_id']}")
    print(f"{'─'*60}")

    session_id = f"scenario_{scenario_num}_{int(time.time())}"
    result = run_query(
        query=scenario["query"],
        customer_id=scenario["customer_id"],
        session_id=session_id,
    )

    print(f"\n{result['final_response']}")
    print(f"\n  Duration: {result['duration_s']}s | Escalated: {result['escalated']}")
    print(f"  Conflicts detected: {len(result['conflicts'])}")
    print(f"  Agents used: {', '.join(result['agents_used'])}")

    return {
        "scenario_number": scenario_num,
        "scenario_name": scenario["name"],
        "query": scenario["query"],
        "customer_id": scenario["customer_id"],
        "session_id": result["session_id"],
        "final_response": result["final_response"],
        "agents_used": result["agents_used"],
        "duration_s": result["duration_s"],
        "escalated": result["escalated"],
        "ticket_id": result.get("ticket_id"),
        "conflicts": result["conflicts"],
        "expected_escalation": scenario["expected_escalation"],
        "escalation_correct": result["escalated"] == scenario["expected_escalation"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="TechCorp Multi-Agent Customer Support System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "How do I enable dark mode?"
  python main.py --scenario 1
  python main.py --scenario 4
  python main.py --scenario all
  python main.py --list-scenarios
        """,
    )
    parser.add_argument("--query",    type=str, help="Run a custom query")
    parser.add_argument("--customer", type=str, default=None, help="Customer ID (e.g. CUST-001)")
    parser.add_argument("--scenario", type=str, help="Scenario number (1-5) or 'all'")
    parser.add_argument("--list-scenarios", action="store_true", help="List all test scenarios")

    args = parser.parse_args()
    print_banner()

    all_results = []

    if args.list_scenarios:
        print("Available Test Scenarios:\n")
        for num, s in TEST_SCENARIOS.items():
            print(f"  [{num}] {s['name']}")
            print(f"       {s['query'][:70]}...")
            print()
        return

    if args.query:
        # Custom query mode
        print(f"\nRunning custom query...")
        result = run_query(args.query, customer_id=args.customer)
        print(f"\n{result['final_response']}")
        all_results.append({
            "scenario_number": 0,
            "scenario_name": "custom",
            "query": args.query,
            **{k: result[k] for k in ["session_id", "final_response", "agents_used",
                                        "duration_s", "escalated", "conflicts"]},
        })

    elif args.scenario:
        if args.scenario.lower() == "all":
            print("\nRunning all 5 test scenarios...\n")
            for num in range(1, 6):
                try:
                    res = run_scenario(num)
                    all_results.append(res)
                    time.sleep(1)  # Brief pause between scenarios
                except Exception as exc:
                    logger.error(f"Scenario {num} failed: {exc}")
                    all_results.append({
                        "scenario_number": num,
                        "error": str(exc),
                    })
        else:
            try:
                num = int(args.scenario)
                if num not in TEST_SCENARIOS:
                    print(f"❌ Invalid scenario number. Choose 1–5.")
                    sys.exit(1)
                res = run_scenario(num)
                all_results.append(res)
            except ValueError:
                print(f"❌ Invalid --scenario value: '{args.scenario}'. Use 1-5 or 'all'.")
                sys.exit(1)
    else:
        parser.print_help()
        return

    if all_results:
        save_results(all_results)

    # Summary
    if len(all_results) > 1:
        print(f"\n{'═'*60}")
        print("  SUMMARY")
        print(f"{'═'*60}")
        for r in all_results:
            status = "✅" if r.get("escalation_correct", True) else "⚠️ "
            print(
                f"  {status} Scenario {r.get('scenario_number')}: "
                f"{r.get('scenario_name', 'custom')} | "
                f"{r.get('duration_s', '?')}s | "
                f"Escalated: {r.get('escalated', '?')}"
            )


if __name__ == "__main__":
    main()
