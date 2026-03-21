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

# ── 1. Session state bootstrap — must be first thing after st import ───────────
def _init_state():
    defaults = {
        "chat_history":      [],
        "current_result":    None,
        "session_id":        str(uuid.uuid4())[:8],
        "total_queries":     0,
        "total_escalations": 0,
        "total_conflicts":   0,
        "running":           False,
        "progress_steps":    [],
        "pending_query":     "",
        "pending_customer":  "",
        "customer_index":    2,   # default to CUST-003 (Enterprise)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── 2. App setup ───────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
Path("results").mkdir(exist_ok=True)
Path("results/sessions").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from monitoring.langfuse_config import get_langfuse_client
    get_langfuse_client()
except Exception as e:
    logger.warning(f"[UI] Langfuse init skipped: {e}")

# ── 3. Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechCorp Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 4. CSS ─────────────────────────────────────────────────────────────────────
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

  .stApp { background: var(--bg-primary); font-family: 'Inter', sans-serif; }
  .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }
  #MainMenu, footer, header { visibility: hidden; }

  [data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border); }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  h1,h2,h3,h4 { color: var(--text-primary) !important; font-weight: 600 !important; }
  p,li,span,label { color: var(--text-secondary) !important; }

  .app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 50%, #161b22 100%);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
  }
  .app-header-title { font-size: 1.6rem; font-weight: 700; color: var(--text-primary) !important; letter-spacing: -0.5px; }
  .app-header-sub   { font-size: 0.8rem; color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace; }

  .metric-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; text-align: center; }
  .metric-val { font-size: 1.6rem; font-weight: 700; color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace; }
  .metric-label { font-size: 0.68rem; color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.8px; }

  .chat-user {
    background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px 10px 2px 10px; padding: 0.9rem 1.1rem; margin: 0.6rem 0;
    color: var(--text-primary) !important;
  }
  .chat-assistant {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px 10px 10px 2px; padding: 0.9rem 1.1rem; margin: 0.6rem 0;
    color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; white-space: pre-wrap; line-height: 1.6;
  }
  .chat-role { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace; }
  .chat-role-user { color: #58a6ff !important; }
  .chat-role-ai   { color: #3fb950 !important; }

  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem; }
  .card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted) !important; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace; }

  .progress-step {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0;
    font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary) !important; border-bottom: 1px solid var(--bg-hover);
  }
  .progress-step:last-child { border-bottom: none; color: var(--accent-green) !important; }

  .agent-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: var(--bg-hover); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 0.73rem;
    color: var(--text-secondary) !important; margin: 3px 2px;
    font-family: 'JetBrains Mono', monospace;
  }
  .agent-chip.active { border-color: var(--accent); color: var(--accent) !important; background: rgba(88,166,255,0.08); }

  .conflict-box {
    background: rgba(240,136,62,0.08); border: 1px solid rgba(240,136,62,0.3);
    border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0;
    font-size: 0.78rem; color: #f0883e !important; font-family: 'JetBrains Mono', monospace;
  }
  .escalation-alert { background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.4); border-radius: 10px; padding: 1rem 1.2rem; margin: 0.6rem 0; }
  .escalation-alert-title { font-size: 0.9rem; font-weight: 700; color: #f85149 !important; margin-bottom: 0.4rem; }

  .stButton > button {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
    transition: all 0.2s !important; width: 100% !important;
  }
  .stButton > button:hover { border-color: var(--accent) !important; background: rgba(88,166,255,0.08) !important; color: var(--accent) !important; }

  .stTextArea textarea {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important; font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important; }

  .stSelectbox > div > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; border-radius: 8px !important; }

  hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

  .output-scroll { max-height: 500px; overflow-y: auto; padding-right: 4px; }
  .output-scroll::-webkit-scrollbar { width: 4px; }
  .output-scroll::-webkit-scrollbar-track { background: var(--bg-secondary); }
  .output-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── 5. Constants ───────────────────────────────────────────────────────────────
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

CUSTOMER_KEYS = list(CUSTOMER_OPTIONS.keys())

AGENT_META = {
    "account":    {"icon": "👤", "label": "Account"},
    "feature":    {"icon": "⚙️",  "label": "Feature"},
    "contract":   {"icon": "📄", "label": "Contract"},
    "escalation": {"icon": "🚨", "label": "Escalation"},
}

# ── 6. Background thread ───────────────────────────────────────────────────────
def _run_query_thread(query: str, customer_id: str, session_id: str, q: Queue):
    """Runs in background thread — never access st.session_state here."""
    try:
        from agents.orchestrator import run_support_crew
        def callback(step):
            q.put(("progress", step))
        result = run_support_crew(
            query=query,
            customer_id=customer_id,
            session_id=session_id,
            progress_callback=callback,
        )
        q.put(("done", result))
    except Exception as e:
        q.put(("error", str(e)))

# ── 7. Render helpers ──────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="app-header">
      <div style="font-size:2.2rem;">🤖</div>
      <div>
        <div class="app-header-title">TechCorp Support Intelligence</div>
        <div class="app-header-sub">CrewAI · Ollama · Langfuse Cloud · 5 Specialized Agents</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Total Queries",      st.session_state.total_queries),
        (c2, "Escalations",        st.session_state.total_escalations),
        (c3, "Conflicts Resolved", st.session_state.total_conflicts),
        (c4, "Session ID",         st.session_state.session_id),
    ]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def render_chat():
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#484f58;">
          <div style="font-size:3rem;margin-bottom:1rem;">💬</div>
          <div style="font-size:1rem;font-weight:500;color:#8b949e;">Ask a support question to begin</div>
          <div style="font-size:0.78rem;margin-top:0.5rem;">Select a test scenario from the sidebar or type below</div>
        </div>""", unsafe_allow_html=True)
        return
    st.markdown('<div class="output-scroll">', unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="chat-role chat-role-user">▸ You</div>{turn["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant"><div class="chat-role chat-role-ai">◈ TechCorp AI</div>{turn["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar_details():
    result = st.session_state.current_result
    if not result:
        return

    st.markdown('<div class="card-title">Last Run — Agents Invoked</div>', unsafe_allow_html=True)
    active = result.get("agents_used", [])
    chips = ""
    for key, meta in AGENT_META.items():
        cls = "agent-chip active" if key in active else "agent-chip"
        chips += f'<span class="{cls}">{meta["icon"]} {meta["label"]}</span>'
    st.markdown(f'<div style="margin-bottom:0.8rem">{chips}</div>', unsafe_allow_html=True)

    dur = result.get("duration_s", 0)
    esc = result.get("escalated", False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{dur}s</div><div class="metric-label">Duration</div></div>', unsafe_allow_html=True)
    with c2:
        color = "#f85149" if esc else "#3fb950"
        label = "YES" if esc else "NO"
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.1rem;color:{color}">{label}</div><div class="metric-label">Escalated</div></div>', unsafe_allow_html=True)

    if esc and result.get("ticket_id"):
        st.markdown(f'<div class="escalation-alert"><div class="escalation-alert-title">🚨 Escalation Ticket</div><div style="font-family:monospace;font-size:0.8rem;color:#f85149">{result["ticket_id"]}</div></div>', unsafe_allow_html=True)

    for c in result.get("conflicts", []):
        st.markdown(f'<div class="conflict-box">{c[:180]}</div>', unsafe_allow_html=True)

    with st.expander("🔬 State JSON", expanded=False):
        st.json(result.get("state_dict", {}))

    agent_outputs = result.get("agent_outputs", {})
    if agent_outputs:
        with st.expander("📋 Raw Agent Outputs", expanded=False):
            for key, output in agent_outputs.items():
                meta = AGENT_META.get(key, {"icon": "•", "label": key.title()})
                st.markdown(f"**{meta['icon']} {meta['label']} Agent**")
                st.code(output[:2000], language="text")
                st.markdown("---")

# ── 8. Main ────────────────────────────────────────────────────────────────────
def main():
    render_header()
    render_metrics()
    st.markdown("<hr>", unsafe_allow_html=True)

    main_col, side_col = st.columns([3, 1], gap="large")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with side_col:
        st.markdown("### 🧪 Test Scenarios")
        for name, data in SCENARIOS.items():
            if st.button(name, key=f"btn_{name}"):
                st.session_state.pending_query    = data["query"]
                st.session_state.pending_customer = data["customer_id"]
                # Resolve index now, before rerun, so no widget conflict
                for i, cid in enumerate(CUSTOMER_OPTIONS.values()):
                    if cid == data["customer_id"]:
                        st.session_state.customer_index = i
                        break
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔬 Last Run Details")
        render_sidebar_details()

    # ── Main column ────────────────────────────────────────────────────────────
    with main_col:
        st.markdown("### 💬 Conversation")
        render_chat()
        st.markdown("<hr>", unsafe_allow_html=True)

        # Customer selector — index driven, never write to widget key
        customer_label = st.selectbox(
            "Customer",
            options=CUSTOMER_KEYS,
            index=st.session_state.customer_index,
        )
        customer_id = CUSTOMER_OPTIONS[customer_label]

        # Pending query from scenario button
        pending_q = st.session_state.pending_query
        if pending_q:
            st.session_state.pending_query = ""

        query_input = st.text_area(
            "Support Query",
            value=pending_q,
            height=90,
            placeholder="Type a support question, or click a scenario above...",
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
                st.session_state.chat_history   = []
                st.session_state.current_result = None
                st.session_state.session_id     = str(uuid.uuid4())[:8]
                st.rerun()

        # ── Run investigation ──────────────────────────────────────────────────
        if send_btn and query_input.strip() and not st.session_state.running:
            query = query_input.strip()
            st.session_state.running = True
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Capture session_id on main thread — threads have no st context
            session_id = st.session_state.session_id
            q: Queue = Queue()
            thread = threading.Thread(
                target=_run_query_thread,
                args=(query, customer_id, session_id, q),
                daemon=True,
            )
            thread.start()

            placeholder = st.empty()
            steps: list[str] = []
            result = None
            error  = None
            t0 = time.time()

            while thread.is_alive() or not q.empty():
                try:
                    kind, payload = q.get(timeout=0.3)
                    if kind == "progress":
                        steps.append(payload)
                        html = "".join(
                            f'<div class="progress-step"><span>{"✅" if i == len(steps)-1 else "⏳"}</span> {s}</div>'
                            for i, s in enumerate(steps)
                        )
                        placeholder.markdown(
                            f'<div class="card"><div class="card-title">🔄 Investigation In Progress</div>{html}</div>',
                            unsafe_allow_html=True,
                        )
                    elif kind == "done":
                        result = payload
                        break
                    elif kind == "error":
                        error = payload
                        break
                except Empty:
                    placeholder.markdown(
                        f'<div class="card"><div class="card-title">⏳ Running... ({time.time()-t0:.0f}s)</div></div>',
                        unsafe_allow_html=True,
                    )

            thread.join(timeout=5)
            placeholder.empty()
            st.session_state.running = False

            if error:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": (
                        f"[ERROR] Investigation failed:\n{error}\n\n"
                        "Please check:\n"
                        "• Ollama is running  (ollama serve)\n"
                        "• Model is pulled    (ollama pull gemma3:12b)\n"
                        "• .env has Langfuse keys"
                    ),
                })
            elif result:
                st.session_state.current_result      = result
                st.session_state.total_queries      += 1
                st.session_state.total_escalations  += int(result.get("escalated", False))
                st.session_state.total_conflicts    += len(result.get("conflicts", []))
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["final_response"],
                })
                # Persist to disk
                try:
                    rp = Path("results/query_results.json")
                    existing = json.loads(rp.read_text()).get("results", []) if rp.exists() else []
                    existing.append({
                        "session_id":     result["session_id"],
                        "timestamp":      datetime.now(timezone.utc).isoformat(),
                        "query":          query,
                        "customer_id":    customer_id,
                        "final_response": result["final_response"],
                        "agents_used":    result["agents_used"],
                        "duration_s":     result["duration_s"],
                        "escalated":      result["escalated"],
                        "ticket_id":      result.get("ticket_id"),
                        "conflicts":      result["conflicts"],
                    })
                    rp.write_text(json.dumps({"results": existing}, indent=2, default=str))
                except Exception as e:
                    logger.warning(f"[UI] Save failed: {e}")

            st.rerun()

    # ── Status bar ─────────────────────────────────────────────────────────────
    try:
        from monitoring.langfuse_config import is_langfuse_enabled
        lf = "🟢 Langfuse Connected" if is_langfuse_enabled() else "🔴 Langfuse Not Configured"
    except Exception:
        lf = "🔴 Langfuse Not Configured"

    st.markdown(f"""
    <div style="position:fixed;bottom:0;left:0;right:0;background:var(--bg-secondary);
                border-top:1px solid var(--border);padding:6px 20px;
                display:flex;justify-content:space-between;align-items:center;
                font-family:'JetBrains Mono',monospace;font-size:0.68rem;z-index:999;">
      <span style="color:var(--text-muted)">TechCorp Support AI</span>
      <span style="color:var(--text-muted)">
        Session: {st.session_state.session_id} &nbsp;|&nbsp; Ollama &nbsp;|&nbsp; {lf}
      </span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
