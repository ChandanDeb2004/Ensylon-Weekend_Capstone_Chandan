"""
state_manager.py
Central shared state for the multi-agent investigation.
All agents read from and write to this state object.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20
MAX_FINDINGS_PER_AGENT = 5


class AgentFinding(BaseModel):
    agent: str
    finding: str
    confidence: str = "high"  # high | medium | low
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tool_calls: list[str] = Field(default_factory=list)


class InvestigationState(BaseModel):
    """Full shared state object passed between agents via CrewAI task context."""

    # ── Session ───────────────────────────────────────────────────────────────
    session_id: str = ""
    original_query: str = ""
    customer_id: Optional[str] = None
    contract_id: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Conversation ──────────────────────────────────────────────────────────
    conversation_history: list[dict] = Field(default_factory=list)

    # ── Agent Findings ────────────────────────────────────────────────────────
    account_findings: list[AgentFinding] = Field(default_factory=list)
    feature_findings: list[AgentFinding] = Field(default_factory=list)
    contract_findings: list[AgentFinding] = Field(default_factory=list)
    escalation_findings: list[AgentFinding] = Field(default_factory=list)

    # ── Decision Tracking ─────────────────────────────────────────────────────
    routing_decisions: list[dict] = Field(default_factory=list)
    conflicts_detected: list[str] = Field(default_factory=list)
    resolution_notes: list[str] = Field(default_factory=list)
    escalation_triggered: bool = False
    escalation_ticket_id: Optional[str] = None
    escalation_priority: Optional[str] = None
    final_decision: Optional[str] = None

    # ── Investigation Progress ────────────────────────────────────────────────
    agents_invoked: list[str] = Field(default_factory=list)
    investigation_complete: bool = False
    failed_tools: list[str] = Field(default_factory=list)
    incomplete_data_notes: list[str] = Field(default_factory=list)

    # ── Context Cache ─────────────────────────────────────────────────────────
    customer_profile: Optional[dict] = None
    contract_data: Optional[dict] = None
    active_plan: Optional[str] = None


class StateManager:
    """
    Manages InvestigationState with pruning to prevent context explosion.
    Serializes state to/from JSON for passing between CrewAI tasks.
    """

    def __init__(self):
        self._state = InvestigationState()

    def initialize(self, session_id: str, query: str, customer_id: Optional[str] = None):
        self._state = InvestigationState(
            session_id=session_id,
            original_query=query,
            customer_id=customer_id,
        )
        self.add_conversation_turn("user", query)
        logger.info(f"[STATE] Initialized session {session_id} | customer={customer_id}")

    # ── Conversation ──────────────────────────────────────────────────────────

    def add_conversation_turn(self, role: str, content: str):
        self._state.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Prune to keep context manageable
        if len(self._state.conversation_history) > MAX_HISTORY_TURNS:
            # Keep first turn (original query) + last N-1 turns
            self._state.conversation_history = (
                self._state.conversation_history[:1] +
                self._state.conversation_history[-(MAX_HISTORY_TURNS - 1):]
            )

    # ── Agent Findings ────────────────────────────────────────────────────────

    def add_finding(self, agent: str, finding: str, confidence: str = "high", tool_calls: list = None):
        entry = AgentFinding(
            agent=agent,
            finding=finding,
            confidence=confidence,
            tool_calls=tool_calls or [],
        )
        store_map = {
            "account":    self._state.account_findings,
            "feature":    self._state.feature_findings,
            "contract":   self._state.contract_findings,
            "escalation": self._state.escalation_findings,
        }
        store = store_map.get(agent.lower(), self._state.account_findings)
        store.append(entry)
        # Prune oldest findings beyond limit
        while len(store) > MAX_FINDINGS_PER_AGENT:
            store.pop(0)
        logger.info(f"[STATE] Finding added | agent={agent} | confidence={confidence}")

    # ── Routing & Decisions ───────────────────────────────────────────────────

    def log_routing_decision(self, from_agent: str, to_agent: str, reason: str):
        self._state.routing_decisions.append({
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })
        if to_agent not in self._state.agents_invoked:
            self._state.agents_invoked.append(to_agent)

    def log_conflict(self, description: str):
        self._state.conflicts_detected.append(description)
        logger.warning(f"[STATE] Conflict detected: {description}")

    def log_resolution(self, note: str):
        self._state.resolution_notes.append(note)

    def log_failed_tool(self, tool_name: str):
        if tool_name not in self._state.failed_tools:
            self._state.failed_tools.append(tool_name)

    def set_escalation(self, ticket_id: str, priority: str):
        self._state.escalation_triggered = True
        self._state.escalation_ticket_id = ticket_id
        self._state.escalation_priority = priority

    def set_final_decision(self, decision: str):
        self._state.final_decision = decision
        self._state.investigation_complete = True

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_context_string(self) -> str:
        """
        Compact representation for injecting into agent prompts.
        Deliberately trimmed to avoid context window overflow.
        """
        s = self._state
        lines = [
            f"SESSION: {s.session_id}",
            f"CUSTOMER_ID: {s.customer_id or 'unknown'}",
            f"CONTRACT_ID: {s.contract_id or 'unknown'}",
            f"ACTIVE_PLAN: {s.active_plan or 'unknown'}",
            f"ORIGINAL_QUERY: {s.original_query}",
            "",
            "── AGENTS INVOKED ──",
            ", ".join(s.agents_invoked) if s.agents_invoked else "none yet",
            "",
        ]

        if s.account_findings:
            lines.append("── ACCOUNT FINDINGS ──")
            for f in s.account_findings[-3:]:  # last 3 only
                lines.append(f"[{f.confidence.upper()}] {f.finding}")

        if s.feature_findings:
            lines.append("── FEATURE FINDINGS ──")
            for f in s.feature_findings[-3:]:
                lines.append(f"[{f.confidence.upper()}] {f.finding}")

        if s.contract_findings:
            lines.append("── CONTRACT FINDINGS ──")
            for f in s.contract_findings[-3:]:
                lines.append(f"[{f.confidence.upper()}] {f.finding}")

        if s.escalation_findings:
            lines.append("── ESCALATION FINDINGS ──")
            for f in s.escalation_findings[-3:]:
                lines.append(f"[{f.confidence.upper()}] {f.finding}")

        if s.conflicts_detected:
            lines.append("── CONFLICTS DETECTED ──")
            for c in s.conflicts_detected:
                lines.append(f"  ⚠ {c}")

        if s.resolution_notes:
            lines.append("── RESOLUTION NOTES ──")
            for r in s.resolution_notes:
                lines.append(f"  ✓ {r}")

        if s.failed_tools:
            lines.append(f"── FAILED TOOLS ──")
            lines.append(", ".join(s.failed_tools))

        if s.escalation_triggered:
            lines.append(f"── ESCALATION ──")
            lines.append(f"Ticket: {s.escalation_ticket_id} | Priority: {s.escalation_priority}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return self._state.model_dump()

    def to_json(self) -> str:
        return self._state.model_dump_json(indent=2)

    @property
    def state(self) -> InvestigationState:
        return self._state
