# TechCorp Multi-Agent Support System — Analysis

## Architecture Summary

This system implements a **CrewAI-based multi-agent pipeline** with 5 specialized agents,
each with distinct tools, system prompts, and investigation responsibilities. All LLM calls
go through **qwen2.5 running locally via Ollama**, and all traces are sent to **Langfuse Cloud**.

---

## Key Design Decisions

### Why CrewAI over LangGraph?
CrewAI was chosen for this project because:
- **Task-based orchestration** maps naturally to the sequential investigation flow
- **Agent role + backstory + goal** system produces better LLM adherence to specialized behavior
- **Sequential Process** with shared task context is simpler to debug than explicit graph edges
- LangGraph would be preferable if we needed complex conditional branching with dynamic looping;
  for this use case, CrewAI's structure is more readable and maintainable

### Query Analysis (Rule-Based Routing)
The Orchestrator uses a **lightweight keyword classifier** rather than an LLM call to determine
which agents to invoke. This keeps routing deterministic, fast, and observable. The routing
decision is logged to Langfuse as a `decision_point` event.

### Memory Architecture
Memory is managed externally (not via CrewAI's built-in memory) through three layers:
1. **StateManager** — structured Pydantic model with per-agent findings, conflict log, routing decisions
2. **ConversationMemory** — disk-persisted turn history for multi-session continuity
3. **SharedContext** — thread-safe in-memory key-value store for cross-agent data within a run

State is pruned before injection into task context to prevent context window overflow.

### Conflict Resolution
Conflicts are detected post-run by scanning agent outputs for known contradiction patterns:
- Documentation claims "unlimited" API but contract/feature shows "1000/month" → resolved by
  treating contract + feature matrix as authoritative (DOC-1182 is the known documentation bug)
- Account plan ≠ contract plan → contract wins
- Seat overage → flagged for awareness

### Tool Resilience
Every tool is wrapped with `@with_tool_behavior` which:
- Injects 100–800ms simulated latency
- Randomly fails 8% of calls (simulating DB timeouts, network errors)
- Returns structured error dicts instead of throwing exceptions
- Logs all calls (tool name, latency, success/failure) for Langfuse

### Escalation Logic
The Escalation Agent applies explicit criteria derived from the problem statement:
- SLA breach on guaranteed contract → CRITICAL (P0)
- Revenue impact >$100/day → HIGH (P1)
- Account suspended + disputed → HIGH
- Waited >2 business days → HIGH
- Confirmed production bug → HIGH

---

## Scenario Execution Walkthrough

### Scenario 1 — Dark Mode (Simple Feature)
`needs_feature=True` only → Feature Agent invoked → retrieves dark_mode documentation →
Enterprise-only, provides setup steps → Escalation Agent: no triggers met → resolved

### Scenario 2 — API on Starter Plan
`needs_account=True`, `needs_feature=True` → Account Agent confirms Starter plan →
Feature Agent confirms API not on Starter, recommends Pro upgrade → no escalation

### Scenario 3 — API Rate Limit Contradiction (High Complexity)
`needs_account=True`, `needs_feature=True`, `needs_contract=True` → All three agents run →
Conflict detected: docs say unlimited, feature matrix + contract say 1000/month →
Conflict resolved: contract + system are authoritative, documentation error DOC-1182 noted →
No escalation (explanation provided)

### Scenario 4 — SLA Violation (Critical Escalation)
`needs_contract=True`, `needs_escalation=True`, `revenue_impact=True` →
Account Agent + Contract Agent + Escalation Agent invoked →
SLA validated: 24-hour guarantee, 240 hours elapsed → breach confirmed →
Escalation triggered: P0 Critical, Customer Success team, immediate notification

### Scenario 5 — Seat Overage / Migration
`needs_account=True`, `needs_feature=True` → Account Agent detects 15 active / 10 licensed →
Feature Agent explains seat licensing and upgrade path →
Escalation Agent: Onboarding team recommended for migration assistance

---

## Known Limitations

1. **Customer ID detection** — system defaults to CUST-003 when no ID is in the query.
   Production would require auth-based customer identification.

2. **LLM determinism** — qwen2.5 outputs can vary; structured output parsing uses regex
   heuristics which may miss edge cases. A more robust approach would use structured output
   schemas (JSON mode).

3. **Parallel execution** — all agents run sequentially. Some tasks (account + feature)
   could run in parallel for speed. CrewAI's `Process.hierarchical` or `asyncio` could enable this.

4. **Tool failure recovery** — current strategy logs failures and continues with partial data.
   Retry logic (tenacity) is available but not implemented per-tool to keep latency predictable.

5. **Context window** — StateManager prunes to last 3 findings per agent and last 20 turns.
   Very long conversations may lose early context. A vector store (e.g. ChromaDB) would
   improve retrieval for longer sessions.

---

## What I Would Improve

- Add **structured output validation** using Pydantic with instructor or JSON mode
- Implement **parallel agent execution** for independent agents (account + feature)
- Add **retry logic** with exponential backoff for tool failures
- Build a **vector memory layer** for semantic retrieval across long conversations
- Add **Langfuse scores** (automated quality scoring per resolved ticket)
- Create a **test suite** with mocked LLM responses for deterministic CI testing
