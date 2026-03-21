# TechCorp Multi-Agent Customer Support Intelligence System

> **Stack:** CrewAI · Gemma 3(Ollama) · Langfuse Cloud · Streamlit

A production-grade multi-agent AI system that handles complex customer support queries
through 5 specialized agents, shared memory, conflict resolution, and SLA-aware escalation.

---

## Architecture

```
Customer Query
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│  • Keyword-based query analysis                         │
│  • Agent routing plan                                   │
│  • Conflict detection + synthesis                       │
└──────────┬──────────────┬────────────┬──────────────────┘
           │              │            │
    ┌──────▼──┐    ┌──────▼──┐  ┌─────▼────┐
    │ Account │    │ Feature │  │ Contract │
    │  Agent  │    │  Agent  │  │  Agent   │
    └──────┬──┘    └──────┬──┘  └─────┬────┘
           │              │            │
           └──────────────▼────────────┘
                          │
                   ┌──────▼──────┐
                   │  Escalation │
                   │    Agent    │
                   └─────────────┘
                          │
                   ┌──────▼──────┐
                   │   Response  │
                   │  Synthesis  │
                   └─────────────┘
```

### Agents

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Orchestrator** | Routes query, detects conflicts, synthesizes response | (analysis logic) |
| **Account Agent** | Plan, seats, billing, enabled features | `lookup_customer`, `get_billing_history`, `check_account_status`, `list_enabled_features` |
| **Feature Agent** | Feature availability, docs, config, limits | `get_feature_matrix`, `get_feature_documentation`, `validate_configuration`, `check_feature_limits` |
| **Contract Agent** | Contract review, SLA validation, entitlements | `lookup_contract`, `get_contract_terms`, `validate_sla_compliance`, `get_included_features` |
| **Escalation Agent** | Escalation decision, ticket creation, routing | `create_escalation_ticket`, `get_escalation_routing`, `notify_support_team`, `log_escalation_reason` |

---

## Setup

### 1. Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Gemma 3 model
ollama pull gemma3:12b

# Verify Ollama is running
ollama serve  # (or it auto-starts as a service)
```

### 2. Clone & Install

```bash
cd techcorp_support
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your **Langfuse Cloud** credentials (from [cloud.langfuse.com](https://cloud.langfuse.com)):

```env
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
```

> ⚠️ If you don't have Langfuse keys, the system runs without tracing (graceful fallback).

---

## Running the System

### Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### CLI — Custom Query

```bash
python main.py --query "How do I enable dark mode?" --customer CUST-003
```

### CLI — Run a Test Scenario

```bash
python main.py --scenario 1    # Basic feature question
python main.py --scenario 4    # SLA violation (triggers escalation)
python main.py --scenario all  # Run all 5 scenarios, save to results/
```

### List Available Scenarios

```bash
python main.py --list-scenarios
```

---

## Project Structure

```
techcorp_support/
├── main.py                   # CLI entry point
├── streamlit_app.py          # Streamlit UI
├── config.yaml               # LLM and system configuration
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── base_agent.py         # LLM factory + Agent builder
│   ├── orchestrator.py       # Routing, crew assembly, synthesis
│   ├── account_agent.py      # Account investigation
│   ├── feature_agent.py      # Feature availability + docs
│   ├── contract_agent.py     # Contract + SLA validation
│   └── escalation_agent.py   # Escalation decision + ticketing
│
├── tools/
│   ├── tool_base.py          # Latency/failure decorator + formatter
│   ├── account_tools.py      # Mock customer database
│   ├── feature_tools.py      # Mock feature matrix
│   ├── contract_tools.py     # Mock contract service
│   └── escalation_tools.py   # Mock ticketing service
│
├── memory/
│   ├── state_manager.py      # Shared investigation state (Pydantic)
│   ├── conversation_memory.py# Disk-persisted conversation history
│   └── shared_context.py     # Thread-safe cross-agent context
│
├── monitoring/
│   ├── langfuse_config.py    # Langfuse Cloud setup + callbacks
│   ├── tracing_utils.py      # Decision/conflict/escalation event logging
│   └── metrics.py            # Per-run metrics recording
│
└── results/
    ├── query_results.json    # Generated by running scenarios
    ├── analysis.md           # Architecture analysis
    └── sessions/             # Per-session conversation history
```

---

## Test Scenarios

| # | Scenario | Agents | Escalates |
|---|----------|--------|-----------|
| 1 | Dark mode setup | Feature, Escalation | No |
| 2 | API access on Starter plan | Account, Feature, Escalation | No |
| 3 | API rate limit contradiction | Account, Feature, Contract, Escalation | No |
| 4 | SLA violation ($500/day impact) | Account, Contract, Escalation | **Yes (Critical)** |
| 5 | Seat overage after migration | Account, Feature, Escalation | No |

---

## Langfuse Observability

The system traces the following events to Langfuse Cloud:

- **Agent calls** — every LLM call via the callback handler
- **Tool invocations** — logged by `@with_tool_behavior` decorator
- **Decision points** — `investigation_plan`, `crew_kickoff` events
- **Conflicts** — `conflict_detected` warning events
- **Escalations** — `escalation_triggered` events with ticket ID and priority
- **Metrics** — `run_metrics` events with duration, agent count, failure count

View traces at: [cloud.langfuse.com](https://cloud.langfuse.com)

---

## Mock Customer Data

| Customer ID | Company | Plan | Notes |
|-------------|---------|------|-------|
| CUST-001 | Acme Corp | Starter | 10 seats licensed, 15 active (overage) |
| CUST-002 | Globex Inc | Pro | API limit contradiction case |
| CUST-003 | Initech LLC | Enterprise | SLA violation case, 24hr guarantee |
| CUST-004 | Umbrella Ltd | Pro | Account suspended |
