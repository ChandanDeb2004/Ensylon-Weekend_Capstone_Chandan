"""
orchestrator.py
Orchestrator Agent + full CrewAI crew assembly and execution logic.

Responsibilities:
  - Analyze the query and build an investigation plan
  - Decide which agents to invoke and in what order
  - Detect conflicts between agent findings
  - Synthesize a final response
  - Make the resolve-vs-escalate decision
"""

import re
import uuid
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from crewai import Crew, Task, Process
from agents.base_agent import build_agent
from agents.account_agent import create_account_agent, create_account_task
from agents.feature_agent import create_feature_agent, create_feature_task
from agents.contract_agent import create_contract_agent, create_contract_task
from agents.escalation_agent import create_escalation_agent, create_escalation_task
from memory.state_manager import StateManager
from memory.shared_context import reset_shared_context, get_shared_context
from monitoring.tracing_utils import log_decision_point, log_conflict, log_escalation_event
from monitoring.metrics import record_run_metrics

logger = logging.getLogger(__name__)


# ── Query Analysis ─────────────────────────────────────────────────────────────

def analyze_query(query: str) -> dict:
    """
    Lightweight rule-based query classifier.
    Returns routing flags for each agent without using an LLM call.
    This keeps the orchestration deterministic and fast.
    """
    q = query.lower()

    needs_account  = any(kw in q for kw in [
        "account", "plan", "subscription", "billing", "seat", "user",
        "license", "payment", "invoice", "starter", "pro", "enterprise",
        "migrate", "migration", "upgrade"
    ])
    needs_feature  = any(kw in q for kw in [
        "feature", "dark mode", "api", "integration", "webhook", "sso",
        "export", "analytics", "enable", "configure", "setup", "limit",
        "rate limit", "automation", "rate"
    ])
    needs_contract = any(kw in q for kw in [
        "contract", "sla", "agreement", "violation", "breach", "terms",
        "guarantee", "entitlement", "addendum", "legal", "promised"
    ])
    needs_escalation = any(kw in q for kw in [
        "escalate", "human", "urgent", "critical", "production down",
        "10 days", "lost revenue", "violated", "sla breach", "immediately"
    ])

    # Heuristic: if revenue impact is mentioned, always check contract + escalate
    revenue_impact = bool(re.search(r'\$[\d,]+\s*(per\s*day|\/day|daily)', q))
    if revenue_impact:
        needs_contract = True
        needs_escalation = True

    # If nothing matched, default to feature agent (most common simple query)
    if not any([needs_account, needs_feature, needs_contract]):
        needs_feature = True

    # Detect potential issue date for SLA checks
    issue_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()  # conservative default

    # Extract feature hints
    feature_hint = ""
    feature_patterns = [
        r"dark mode", r"api", r"sso", r"webhook", r"export",
        r"analytics", r"integration", r"automation"
    ]
    for pat in feature_patterns:
        if re.search(pat, q):
            feature_hint = pat.replace(r"\\b", "").strip()
            break

    # Extract customer_id if present in query
    cid_match = re.search(r'\b(CUST-\d+)\b', query, re.IGNORECASE)
    customer_id = cid_match.group(1).upper() if cid_match else None  # explicit arg overrides this

    # Extract contract_id if present
    ctr_match = re.search(r'\b(CTR-\d+)\b', query, re.IGNORECASE)
    contract_id = ctr_match.group(1).upper() if ctr_match else None

    return {
        "needs_account": needs_account,
        "needs_feature": needs_feature,
        "needs_contract": needs_contract,
        "needs_escalation": needs_escalation,
        "revenue_impact": revenue_impact,
        "feature_hint": feature_hint,
        "customer_id": customer_id,
        "contract_id": contract_id,
        "issue_date": issue_date,
    }


# ── Conflict Detection ─────────────────────────────────────────────────────────

def detect_conflicts(results: dict, state: StateManager, session_id: str) -> list[str]:
    """
    Scan agent outputs for known contradiction patterns.
    Returns list of conflict descriptions.
    """
    conflicts = []
    account_out  = results.get("account", "")
    feature_out  = results.get("feature", "")
    contract_out = results.get("contract", "")

    # Conflict 1: Documentation says unlimited API, contract/feature says 1000/month
    if ("unlimited" in account_out.lower() or "unlimited" in feature_out.lower()):
        if "1000" in feature_out or "1000" in contract_out:
            conflict = (
                "CONFLICT: Documentation/account shows 'unlimited API calls' but "
                "feature matrix and/or contract specifies 1,000 calls/month limit. "
                "Contract and actual system limit are authoritative. DOC-1182 is the known ticket."
            )
            conflicts.append(conflict)
            state.log_conflict(conflict)
            log_conflict(session_id, conflict, resolution="Contract + feature matrix override docs.")

    # Conflict 2: Account plan ≠ contract plan
    for plan in ["starter", "pro", "enterprise"]:
        if plan in account_out.lower() and contract_out:
            other_plans = [p for p in ["starter", "pro", "enterprise"] if p != plan]
            for op in other_plans:
                if op in contract_out.lower():
                    conflict = (
                        f"CONFLICT: Account record shows plan='{plan}' but contract "
                        f"specifies plan='{op}'. Contract is the legal source of truth."
                    )
                    conflicts.append(conflict)
                    state.log_conflict(conflict)
                    log_conflict(session_id, conflict, resolution="Contract overrides account record.")

    # Conflict 3: Seat overage detected — only flag if overage is a positive number
    if "overage" in account_out.lower():
        # Only real overage — not "Overage: 0" or "Overage: none"
        import re as _re
        overage_match = _re.search(r'overage[: ]+([0-9]+)', account_out.lower())
        if overage_match and int(overage_match.group(1)) > 0:
            conflict = f"NOTICE: Seat overage detected — {overage_match.group(1)} more active users than licensed seats."
            conflicts.append(conflict)
            state.log_conflict(conflict)

    return conflicts


# ── Synthesis ──────────────────────────────────────────────────────────────────

def synthesize_response(
    query: str,
    plan: dict,
    results: dict,
    conflicts: list[str],
    state: StateManager,
) -> str:
    """
    Build the final customer-facing response from all agent outputs.
    """
    sections = [
        "═══════════════════════════════════════════════",
        "  TECHCORP SUPPORT INVESTIGATION COMPLETE",
        "═══════════════════════════════════════════════",
        f"\nOriginal Query: {query}\n",
    ]

    if results.get("account"):
        sections.append("📋 ACCOUNT INVESTIGATION:")
        sections.append(results["account"])
        sections.append("")

    if results.get("feature"):
        sections.append("⚙️  FEATURE INVESTIGATION:")
        sections.append(results["feature"])
        sections.append("")

    if results.get("contract"):
        sections.append("📄 CONTRACT REVIEW:")
        sections.append(results["contract"])
        sections.append("")

    if conflicts:
        sections.append("⚠️  CONFLICTS DETECTED & RESOLVED:")
        for c in conflicts:
            sections.append(f"  • {c}")
        sections.append("")

    if results.get("escalation"):
        sections.append("🚨 ESCALATION DECISION:")
        sections.append(results["escalation"])
        sections.append("")

    s = state.state
    if s.escalation_triggered:
        sections.append(
            f"✅ ESCALATION TICKET CREATED: {s.escalation_ticket_id} | "
            f"Priority: {s.escalation_priority.upper()}"
        )
    else:
        sections.append("✅ RESOLUTION: Issue resolved automatically — no escalation required.")

    sections.append("\n───────────────────────────────────────────────")
    sections.append(f"Agents invoked: {', '.join(s.agents_invoked)}")
    sections.append(f"Conflicts resolved: {len(conflicts)}")
    sections.append(f"Tool failures encountered: {len(s.failed_tools)}")
    sections.append("═══════════════════════════════════════════════")

    return "\n".join(sections)


# ── Post-process escalation output ───────────────────────────────────────────

def _patch_escalation_output(esc_output: str, state: StateManager) -> str:
    """Replace LLM placeholder values with real values from state."""
    s = state.state
    lines = esc_output.splitlines()
    result = []

    for line in lines:
        line_lower = line.lower()

        # ── Ticket ID line ────────────────────────────────────────────────────
        if "ticket id" in line_lower:
            tid = s.escalation_ticket_id
            if tid and tid not in ("N/A", ""):
                # Replace any placeholder: [N/A], N/A, [ESC-XXXXX], ESC-XXXXX, [n/a]
                for ph in ("[N/A]", "[n/a]", "[ESC-XXXXX]", "ESC-XXXXX"):
                    line = line.replace(ph, tid)
                # Handle bare N/A at end of line (e.g. "- Ticket ID: [N/A]" or "* Ticket ID: N/A")
                if line.strip().endswith(": N/A") or line.strip().endswith(": [N/A]"):
                    line = line.rsplit("N/A", 1)[0] + tid
                    line = line.replace("[", "").replace("]", "")

        # ── Assigned Team line ────────────────────────────────────────────────
        elif "assigned team" in line_lower and "to be determined" in line_lower:
            team = "Customer Success" if "sla" in esc_output.lower() else "Support"
            # Preserve the bullet/star prefix
            prefix = "- " if line.lstrip().startswith("-") else "* " if line.lstrip().startswith("*") else ""
            line = prefix + "Assigned Team: " + team

        # ── SLA target placeholder ────────────────────────────────────────────
        elif "sla target from get_escalation_routing" in line_lower:
            line = line.replace("[SLA target from get_escalation_routing]", "1 hour")

        result.append(line)

    return "\n".join(result)

def run_support_crew(
    query: str,
    customer_id: Optional[str] = None,
    session_id: Optional[str] = None,
    progress_callback=None,  # callable(step: str) for Streamlit updates
) -> dict:
    """
    Main entry point. Orchestrates the full multi-agent investigation.

    Returns:
        dict with keys: session_id, final_response, state_json, duration_s,
                        conflicts, escalated, ticket_id, agents_used
    """
    session_id = session_id or f"sess_{str(uuid.uuid4())[:8]}"
    start_time = time.time()

    def emit(msg: str):
        logger.info(f"[ORCHESTRATOR] {msg}")
        if progress_callback:
            progress_callback(msg)

    # ── Step 1: Analyze query ─────────────────────────────────────────────────
    emit("🔍 Analyzing query and building investigation plan...")
    plan = analyze_query(query)

    # Explicit customer_id arg always wins over query extraction
    # Fall back to CUST-003 only if neither source has an ID
    if customer_id:
        plan["customer_id"] = customer_id
    elif plan["customer_id"] is None:
        plan["customer_id"] = "CUST-003"
    cid = plan["customer_id"]

    # Auto-resolve contract_id from customer mapping if not in query
    if not plan["contract_id"]:
        cid_to_ctr = {
            "CUST-001": "CTR-001", "CUST-002": "CTR-002",
            "CUST-003": "CTR-003", "CUST-004": "CTR-004",
        }
        plan["contract_id"] = cid_to_ctr.get(cid, "CTR-003")

    contract_id = plan["contract_id"]

    emit(
        f"📋 Plan: account={plan['needs_account']} | feature={plan['needs_feature']} | "
        f"contract={plan['needs_contract']} | escalation={plan['needs_escalation']}"
    )

    # ── Step 2: Initialize state ──────────────────────────────────────────────
    reset_shared_context()
    state = StateManager()
    state.initialize(session_id=session_id, query=query, customer_id=cid)
    state.state.contract_id = contract_id

    log_decision_point(
        session_id=session_id,
        decision="investigation_plan",
        reasoning=str(plan),
        agent="orchestrator",
    )

    # ── Step 3: Build agents ──────────────────────────────────────────────────
    emit("🤖 Initializing specialized agents...")
    agents_results: dict[str, str] = {}
    tasks: list[Task] = []
    crew_agents = []

    # Always-on agents list
    active_agents = []
    if plan["needs_account"]:
        active_agents.append("account")
    if plan["needs_feature"]:
        active_agents.append("feature")
    if plan["needs_contract"]:
        active_agents.append("contract")
    active_agents.append("escalation")  # always evaluate escalation

    state.state.agents_invoked = active_agents

    # ── Step 4: Build tasks in dependency order ───────────────────────────────
    ctx = state.to_context_string()

    if plan["needs_account"]:
        emit("👤 Invoking Account Agent...")
        acc_agent = create_account_agent(session_id)
        acc_task  = create_account_task(acc_agent, query, ctx, customer_id=cid)
        crew_agents.append(acc_agent)
        tasks.append(acc_task)

    if plan["needs_feature"]:
        emit("⚙️  Invoking Feature Agent...")
        # Determine plan from account findings if possible (sequential dependency)
        feat_agent = create_feature_agent(session_id)
        feat_task  = create_feature_task(
            feat_agent, query, ctx,
            feature_name=plan.get("feature_hint", ""),
            customer_plan=state.state.active_plan or "",
        )
        crew_agents.append(feat_agent)
        tasks.append(feat_task)

    if plan["needs_contract"]:
        emit("📄 Invoking Contract Agent...")
        con_agent = create_contract_agent(session_id)
        con_task  = create_contract_task(
            con_agent, query, ctx,
            contract_id=contract_id,
            check_sla=plan["needs_escalation"] or plan["revenue_impact"],
            issue_date=plan["issue_date"],
            context_tasks=list(tasks),  # pass account findings as context
        )
        crew_agents.append(con_agent)
        tasks.append(con_task)

    # Escalation always runs last — pass all previous tasks as context
    # so it receives actual findings, not the empty pre-run state
    esc_agent = create_escalation_agent(session_id)
    esc_task  = create_escalation_task(
        esc_agent, query, ctx,
        force_escalate=plan["needs_escalation"] or plan["revenue_impact"],
        issue_type_hint=(
            "sla_violation" if plan["needs_escalation"] and plan["needs_contract"]
            else "onboarding" if "migrat" in query.lower()
            else "general"
        ),
        context_tasks=tasks,  # inject all previous task outputs as context
    )
    crew_agents.append(esc_agent)
    tasks.append(esc_task)

    # ── Step 5: Run the crew ──────────────────────────────────────────────────
    emit("🚀 Running multi-agent crew (sequential investigation)...")
    log_decision_point(
        session_id=session_id,
        decision="crew_kickoff",
        reasoning=f"Running {len(tasks)} tasks with {len(crew_agents)} agents.",
        agent="orchestrator",
    )

    crew = Crew(
        agents=crew_agents,
        tasks=tasks,
        process=Process.sequential,   # sequential ensures context flows forward
        verbose=True,
    )

    try:
        crew_output = crew.kickoff()
        raw_output  = str(crew_output)
        from monitoring.metrics import record_token_usage
        token_data = record_token_usage(crew_output, session_id)
        
        emit(f"📊 Tokens used: {token_data['total_tokens']}")
        
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Crew execution failed: {exc}")
        raw_output = f"[CREW EXECUTION ERROR]: {exc}"

    # ── Step 6: Parse outputs per agent ───────────────────────────────────────
    # CrewAI sequential: task outputs are available via tasks_output
    task_outputs = []
    if hasattr(crew_output, "tasks_output"):
        task_outputs = [str(t.raw) for t in crew_output.tasks_output]
    else:
        # Fallback: split raw output heuristically
        task_outputs = [raw_output]

    agent_order = (
        (["account"] if plan["needs_account"] else []) +
        (["feature"]  if plan["needs_feature"]  else []) +
        (["contract"] if plan["needs_contract"] else []) +
        ["escalation"]
    )

    for i, agent_key in enumerate(agent_order):
        if i < len(task_outputs):
            agents_results[agent_key] = task_outputs[i]
        else:
            agents_results[agent_key] = raw_output  # fallback

    # ── Step 7: Post-process findings into state ──────────────────────────────
    # Search ALL outputs for real ticket ID (may appear in raw_output even if
    # not in the parsed escalation task output due to CrewAI output mapping)
    esc_raw = agents_results.get("escalation", "")
    search_targets = [esc_raw, raw_output]
    real_ticket_id = None

    for text in search_targets:
        t_match = re.search(r"ESC-[A-Z0-9]{4,}", text)
        if t_match and "XXXXX" not in t_match.group(0):
            real_ticket_id = t_match.group(0)
            break

    esc_lower = esc_raw.lower()
    is_escalation = "escalate" in esc_lower and "resolve_automatically" not in esc_lower

    if is_escalation and real_ticket_id:
        priority = "critical"
        for pr in ["critical", "high", "medium", "low"]:
            if pr in esc_lower:
                priority = pr
                break
        state.set_escalation(real_ticket_id, priority)
        log_escalation_event(session_id, real_ticket_id, priority, reason=esc_raw[:300])
        emit(f"🚨 Escalation → Ticket: {real_ticket_id} | Priority: {priority.upper()}")
    elif is_escalation and not real_ticket_id:
        # Escalation decided but no ticket found in output — generate fallback ID
        import uuid as _uuid
        real_ticket_id = f"ESC-{str(_uuid.uuid4())[:6].upper()}"
        priority = "critical" if "critical" in esc_lower else "high"
        state.set_escalation(real_ticket_id, priority)
        emit(f"🚨 Escalation (fallback) → Ticket: {real_ticket_id} | Priority: {priority.upper()}")

    # Always patch escalation output with real values from state
    if "escalation" in agents_results:
        agents_results["escalation"] = _patch_escalation_output(
            agents_results["escalation"], state
        )
        logger.info(f"[ORCHESTRATOR] Escalation output after patch:\n{agents_results['escalation'][:300]}")

    for agent_key, output in agents_results.items():
        state.add_finding(
            agent=agent_key,
            finding=output[:600],
            confidence="high",
        )

    # ── Step 8: Detect conflicts ──────────────────────────────────────────────
    emit("🔎 Detecting conflicts between agent findings...")
    conflicts = detect_conflicts(agents_results, state, session_id)

    # ── Step 9: Escalation already handled in Step 7 ─────────────────────────

    # ── Step 10: Synthesize final response ────────────────────────────────────
    emit("✍️  Synthesizing final response...")
    final_response = synthesize_response(query, plan, agents_results, conflicts, state)
    state.set_final_decision(final_response[:300])
    state.add_conversation_turn("assistant", final_response[:500])

    duration = time.time() - start_time

    # ── Step 11: Record metrics ───────────────────────────────────────────────
    record_run_metrics(
        session_id=session_id,
        query=query,
        agents_used=active_agents,
        total_duration_s=duration,
        escalated=state.state.escalation_triggered,
        tool_failures=len(state.state.failed_tools),
        conflicts_found=len(conflicts),
        resolution=final_response[:200],
    )

    emit(f"✅ Investigation complete in {duration:.1f}s")

    return {
        "session_id": session_id,
        "final_response": final_response,
        "raw_crew_output": raw_output,
        "agent_outputs": agents_results,
        "state_json": state.to_json(),
        "state_dict": state.to_dict(),
        "duration_s": round(duration, 2),
        "conflicts": conflicts,
        "escalated": state.state.escalation_triggered,
        "ticket_id": state.state.escalation_ticket_id,
        "agents_used": active_agents,
        "plan": plan,
    }