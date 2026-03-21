"""
streamlit_app.py
TechCorp Support Intelligence System — Streamlit Front-End

Run: streamlit run streamlit_app.py
"""

import sys
import time
import json
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from queue import Queue, Empty

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

Path("results").mkdir(exist_ok=True)
Path("results/sessions").mkdir(exist_ok=True)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechCorp Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --bg-hover:      #21262d;
    --border:        #30363d;
    --accent:        #58a6ff;
    --accent-green:  #3fb950;
    --accent-orange: #f0883e;
    --accent-red:    #f85149;
    --accent-purple: #bc8cff;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
  }

  /* Base */
  .stApp { background: var(--bg-primary); font-family: 'Inter', sans-serif; }
  .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  /* Typography */
  h1, h2, h3, h4 { color: var(--text-primary) !important; font-weight: 600 !important; }
  p, li, span, label { color: var(--text-secondary) !important; }

  /* Header banner */
  .app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 50%, #161b22 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .app-header-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.5px;
  }
  .app-header-sub {
    font-size: 0.8rem;
    color: var(--text-muted) !important;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Status badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
  }
  .badge-blue   { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
  .badge-green  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
  .badge-red    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
  .badge-orange { background: rgba(240,136,62,0.15); color: #f0883e; border: 1px solid rgba(240,136,62,0.3); }
  .badge-purple { background: rgba(188,140,255,0.15);color: #bc8cff; border: 1px solid rgba(188,140,255,0.3); }

  /* Cards */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted) !important;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Chat messages */
  .chat-user {
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px 10px 2px 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    color: var(--text-primary) !important;
  }
  .chat-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px 10px 10px 2px;
    padding: 0.9rem 1.1rem;
    margin: 0.6rem 0;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    line-height: 1.6;
  }
  .chat-role {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
  }
  .chat-role-user { color: #58a6ff !important; }
  .chat-role-ai   { color: #3fb950 !important; }

  /* Progress steps */
  .progress-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary) !important;
    border-bottom: 1px solid var(--bg-hover);
  }
  .progress-step:last-child { border-bottom: none; color: var(--accent-green) !important; }

  /* Agent badges in sidebar */
  .agent-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-hover);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.73rem;
    color: var(--text-secondary) !important;
    margin: 3px 2px;
    font-family: 'JetBrains Mono', monospace;
  }
  .agent-chip.active {
    border-color: var(--accent);
    color: var(--accent) !important;
    background: rgba(88,166,255,0.08);
  }

  /* Metrics row */
  .metric-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
  }
  .metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-label {
    font-size: 0.68rem;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  /* Conflict box */
  .conflict-box {
    background: rgba(240,136,62,0.08);
    border: 1px solid rgba(240,136,62,0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.78rem;
    color: #f0883e !important;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Escalation alert */
  .escalation-alert {
    background: rgba(248,81,73,0.1);
    border: 1px solid rgba(248,81,73,0.4);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
  }
  .escalation-alert-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #f85149 !important;
    margin-bottom: 0.4rem;
  }

  /* Scenario buttons */
  .stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(88,166,255,0.08) !important;
    color: var(--accent) !important;
  }

  /* Input */
  .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
  }

  .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
  }

  /* Divider */
  hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

  /* Scrollable output */
  .output-scroll {
    max-height: 500px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .output-scroll::-webkit-scrollbar { width: 4px; }
  .output-scroll::-webkit-scrollbar-track { background: var(--bg-secondary); }
  .output-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history": [],
        "current_result": None,
        "session_id": str(uuid.uuid4())[:8],
        "total_queries": 0,
        "total_escalations": 0,
        "total_conflicts": 0,
        "running": False,
        "progress_steps": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ── Test Scenarios ─────────────────────────────────────────────────────────────
SCENARIOS = {
    "1 · Dark Mode Setup": {
        "query": "How do I enable dark mode in my account?",
        "customer_id": "CUST-003",
    },
    "2 · API Access (Starter Plan)": {
        "query": "I'm on the Starter plan, but I need to integrate with your API for my automation workflow. What are my options?",
        "customer_id": "CUST-001",
    },
    "3 · API Rate Limit Contradiction": {
        "query": "Your documentation says the Pro plan includes unlimited API calls, but I'm seeing rate limit errors after 1000 calls/month. I've checked my account and it shows Pro. Is this a bug?",
        "customer_id": "CUST-002",
    },
    "4 · SLA Violation (Critical)": {
        "query": "I've been waiting for support response for 10 days on a critical production issue. My company has a contract with a 24-hour SLA guarantee. This is now costing us $500/day in lost revenue. Please verify if the SLA was violated and escalate this immediately.",
        "customer_id": "CUST-003",
    },
    "5 · Seat License Overage": {
        "query": "Our company just migrated from the competitor platform. We have 15 users, but the plan shows only 10 seats. Can you help me understand the licensing model and figure out how to set up all our users?",
        "customer_id": "CUST-001",
    },
}

CUSTOMER_OPTIONS = {
    "CUST-001 · Acme Corp (Starter)":     "CUST-001",
    "CUST-002 · Globex Inc (Pro)":         "CUST-002",
    "CUST-003 · Initech LLC (Enterprise)": "CUST-003",
    "CUST-004 · Umbrella Ltd (Suspended)": "CUST-004",
}

AGENT_META = {
    "account":    {"icon": "👤", "label": "Account",    "color": "badge-blue"},
    "feature":    {"icon": "⚙️",  "label": "Feature",    "color": "badge-purple"},
    "contract":   {"icon": "📄", "label": "Contract",   "color": "badge-orange"},
    "escalation": {"icon": "🚨", "label": "Escalation", "color": "badge-red"},
}


# ── Run Query (thread-safe) ────────────────────────────────────────────────────
def run_query_threaded(query: str, customer_id: str, progress_queue: Queue):
    """Runs in a background thread — pushes progress updates into a Queue."""
    try:
        from agents.orchestrator import run_support_crew

        def callback(step: str):
            progress_queue.put(("progress", step))

        result = run_support_crew(
            query=query,
            customer_id=customer_id,
            session_id=st.session_state.session_id,
            progress_callback=callback,
        )
        progress_queue.put(("done", result))
    except Exception as exc:
        logger.error(f"[UI] Query failed: {exc}")
        progress_queue.put(("error", str(exc)))


# ── Render Functions ───────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="app-header">
      <div style="font-size:2.2rem;">🤖</div>
      <div>
        <div class="app-header-title">TechCorp Support Intelligence</div>
        <div class="app-header-sub">CrewAI · Gemma 3 · Langfuse Cloud · 5 Specialized Agents</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_row():
    cols = st.columns(4)
    metrics = [
        ("Total Queries",     st.session_state.total_queries,     "badge-blue"),
        ("Escalations",       st.session_state.total_escalations,  "badge-red"),
        ("Conflicts Resolved",st.session_state.total_conflicts,    "badge-orange"),
        ("Session ID",        st.session_state.session_id,         "badge-purple"),
    ]
    for col, (label, val, _badge) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_agent_chips(active_agents: list = None):
    active = active_agents or []
    chips_html = ""
    for key, meta in AGENT_META.items():
        is_active = key in active
        cls = "agent-chip active" if is_active else "agent-chip"
        chips_html += f'<span class="{cls}">{meta["icon"]} {meta["label"]}</span>'
    st.markdown(f'<div style="margin-bottom:0.8rem">{chips_html}</div>', unsafe_allow_html=True)


def render_chat_history():
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #484f58;">
          <div style="font-size:3rem; margin-bottom:1rem;">💬</div>
          <div style="font-size:1rem; font-weight:500; color:#8b949e;">Ask a support question to begin</div>
          <div style="font-size:0.78rem; margin-top:0.5rem;">
            Select a test scenario from the sidebar or type your own query below
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="output-scroll">', unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
              <div class="chat-role chat-role-user">▸ You</div>
              {turn["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-assistant">
              <div class="chat-role chat-role-ai">◈ TechCorp AI</div>
              {turn["content"]}
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_result_sidebar(result: dict):
    if not result:
        return

    st.markdown('<div class="card-title">Last Run — Agents Invoked</div>', unsafe_allow_html=True)
    render_agent_chips(result.get("agents_used", []))

    # Duration + escalation
    dur = result.get("duration_s", 0)
    esc = result.get("escalated", False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-val" style="font-size:1.1rem">{dur}s</div>
          <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        color = "#f85149" if esc else "#3fb950"
        label = "YES" if esc else "NO"
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-val" style="font-size:1.1rem; color:{color}">{label}</div>
          <div class="metric-label">Escalated</div>
        </div>
        """, unsafe_allow_html=True)

    if result.get("escalated") and result.get("ticket_id"):
        st.markdown(f"""
        <div class="escalation-alert">
          <div class="escalation-alert-title">🚨 Escalation Ticket Created</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#f85149;">
            {result["ticket_id"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Conflicts
    conflicts = result.get("conflicts", [])
    if conflicts:
        st.markdown(f'<div class="card-title" style="margin-top:1rem">⚠️ Conflicts Resolved ({len(conflicts)})</div>', unsafe_allow_html=True)
        for c in conflicts:
            st.markdown(f'<div class="conflict-box">{c[:180]}</div>', unsafe_allow_html=True)

    # State inspector
    with st.expander("🔬 Investigation State JSON", expanded=False):
        try:
            state_dict = result.get("state_dict", {})
            st.json(state_dict)
        except Exception:
            st.code(result.get("state_json", "{}"), language="json")

    # Raw agent outputs
    agent_outputs = result.get("agent_outputs", {})
    if agent_outputs:
        with st.expander("📋 Raw Agent Outputs", expanded=False):
            for agent_key, output in agent_outputs.items():
                meta = AGENT_META.get(agent_key, {"icon": "•", "label": agent_key.title()})
                st.markdown(f"**{meta['icon']} {meta['label']} Agent**")
                st.code(output[:2000], language="text")
                st.markdown("---")


# ── Main Layout ────────────────────────────────────────────────────────────────
def main():
    render_header()
    render_metrics_row()
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Layout: main chat + right sidebar ─────────────────────────────────────
    main_col, side_col = st.columns([3, 1], gap="large")

    with side_col:
        st.markdown("### 🧪 Test Scenarios")
        for scenario_name, scenario_data in SCENARIOS.items():
            if st.button(scenario_name, key=f"scn_{scenario_name}"):
                st.session_state["pending_query"]   = scenario_data["query"]
                st.session_state["pending_customer"] = scenario_data["customer_id"]
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔬 Last Run Details")
        render_result_sidebar(st.session_state.current_result)

    with main_col:
        st.markdown("### 💬 Conversation")
        chat_container = st.container()

        with chat_container:
            render_chat_history()

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Query input form ───────────────────────────────────────────────────
        with st.container():
            customer_label = st.selectbox(
                "Customer",
                options=list(CUSTOMER_OPTIONS.keys()),
                index=2,
                key="customer_select",
            )
            customer_id = CUSTOMER_OPTIONS[customer_label]

            # Use pending query from scenario button if set
            default_query = st.session_state.pop("pending_query", "")
            if "pending_customer" in st.session_state:
                pending_cust = st.session_state.pop("pending_customer", None)
                if pending_cust:
                    # Find matching key
                    for lbl, cid in CUSTOMER_OPTIONS.items():
                        if cid == pending_cust:
                            st.session_state["customer_select"] = lbl
                            break

            query_input = st.text_area(
                "Support Query",
                value=default_query,
                height=90,
                placeholder="Type a customer support question, or click a scenario above...",
                key="query_input",
            )

            send_col, clear_col = st.columns([4, 1])
            with send_col:
                send_btn = st.button(
                    "🚀  Run Investigation",
                    disabled=st.session_state.running,
                    type="primary",
                    use_container_width=True,
                )
            with clear_col:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.current_result = None
                    st.session_state.session_id = str(uuid.uuid4())[:8]
                    st.rerun()

        # ── Execute query ──────────────────────────────────────────────────────
        if send_btn and query_input.strip() and not st.session_state.running:
            query = query_input.strip()
            st.session_state.running = True
            st.session_state.progress_steps = []
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Progress tracking
            progress_placeholder = st.empty()
            progress_queue: Queue = Queue()

            # Launch background thread
            thread = threading.Thread(
                target=run_query_threaded,
                args=(query, customer_id, progress_queue),
                daemon=True,
            )
            thread.start()

            # Poll for progress
            steps_seen = []
            result = None
            error_msg = None
            start_time = time.time()

            while thread.is_alive() or not progress_queue.empty():
                try:
                    msg_type, payload = progress_queue.get(timeout=0.3)
                    if msg_type == "progress":
                        steps_seen.append(payload)
                        steps_html = "".join(
                            f'<div class="progress-step"><span>{"✅" if i == len(steps_seen)-1 else "⏳"}</span> {s}</div>'
                            for i, s in enumerate(steps_seen)
                        )
                        progress_placeholder.markdown(
                            f'<div class="card"><div class="card-title">🔄 Investigation In Progress</div>{steps_html}</div>',
                            unsafe_allow_html=True,
                        )
                    elif msg_type == "done":
                        result = payload
                        break
                    elif msg_type == "error":
                        error_msg = payload
                        break
                except Empty:
                    # Still running — show spinner pulse
                    elapsed = time.time() - start_time
                    progress_placeholder.markdown(
                        f'<div class="card"><div class="card-title">⏳ Running... ({elapsed:.0f}s)</div></div>',
                        unsafe_allow_html=True,
                    )

            thread.join(timeout=5)
            progress_placeholder.empty()
            st.session_state.running = False

            if error_msg:
                st.error(f"❌ Investigation failed: {error_msg}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"[ERROR] The investigation encountered an error:\n{error_msg}\n\nPlease check:\n• Ollama is running (ollama serve)\n• Gemma3 model is pulled (ollama pull Gemma 3)\n• .env file has Langfuse keys",
                })
            elif result:
                st.session_state.current_result = result
                st.session_state.total_queries += 1

                if result.get("escalated"):
                    st.session_state.total_escalations += 1
                if result.get("conflicts"):
                    st.session_state.total_conflicts += len(result["conflicts"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["final_response"],
                })

                # Save result to file
                try:
                    results_path = Path("results/query_results.json")
                    existing = []
                    if results_path.exists():
                        with open(results_path) as f:
                            data = json.load(f)
                            existing = data.get("results", [])
                    existing.append({
                        "session_id": result["session_id"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "query": query,
                        "customer_id": customer_id,
                        "final_response": result["final_response"],
                        "agents_used": result["agents_used"],
                        "duration_s": result["duration_s"],
                        "escalated": result["escalated"],
                        "ticket_id": result.get("ticket_id"),
                        "conflicts": result["conflicts"],
                    })
                    with open(results_path, "w") as f:
                        json.dump({"results": existing}, f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"[UI] Could not save result: {e}")

            st.rerun()


    # ── Bottom status bar ──────────────────────────────────────────────────────
    from monitoring.langfuse_config import is_langfuse_enabled
    lf_status = "🟢 Langfuse Connected" if is_langfuse_enabled() else "🔴 Langfuse Not Configured"

    st.markdown(
        f"""
        <div style="position:fixed; bottom:0; left:0; right:0; background:var(--bg-secondary);
                    border-top:1px solid var(--border); padding:6px 20px;
                    display:flex; justify-content:space-between; align-items:center;
                    font-family:'JetBrains Mono',monospace; font-size:0.68rem; z-index:999;">
          <span style="color:var(--text-muted)">TechCorp Support AI</span>
          <span style="color:var(--text-muted)">
            Session: {st.session_state.session_id} &nbsp;|&nbsp;
            gemma3:12b via Ollama &nbsp;|&nbsp;
            {lf_status}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
