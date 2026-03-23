"""
streamlit_app.py
TechCorp Support Intelligence System — Streamlit Front-End
Run: streamlit run streamlit_app.py
"""

import os
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

# ── 1. Session state — must be first ──────────────────────────────────────────
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
        "customer_index":    2,
        # LLM settings
        "llm_provider":      "ollama",
        "llm_model":         "gemma3:12b",
        "llm_api_key":       "",
        "llm_configured":    True,   # ollama needs no key
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
    from monitoring.langfuse_config import configure
    configure()
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

  /* LLM provider pills */
  .provider-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
    border: 1px solid; margin: 2px;
  }
  .provider-ollama  { background: rgba(63,185,80,0.12);  color: #3fb950; border-color: rgba(63,185,80,0.3); }
  .provider-ollama_cloud { background: rgba(63,185,80,0.12); color: #3fb950; border-color: rgba(63,185,80,0.3); }
  .provider-groq    { background: rgba(248,81,73,0.12);  color: #f85149; border-color: rgba(248,81,73,0.3); }
  .provider-gemini  { background: rgba(88,166,255,0.12); color: #58a6ff; border-color: rgba(88,166,255,0.3); }
  .provider-openai  { background: rgba(188,140,255,0.12);color: #bc8cff; border-color: rgba(188,140,255,0.3); }
  .provider-anthropic { background: rgba(240,136,62,0.12);color: #f0883e; border-color: rgba(240,136,62,0.3); }

  .llm-config-box {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
  }

  .stButton > button {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
    transition: all 0.2s !important; width: 100% !important;
  }
  .stButton > button:hover { border-color: var(--accent) !important; background: rgba(88,166,255,0.08) !important; color: var(--accent) !important; }

  .stTextArea textarea, .stTextInput input {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important; font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
  }
  .stSelectbox > div > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; border-radius: 8px !important; }

  hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
  .output-scroll { max-height: 500px; overflow-y: auto; padding-right: 4px; }
  .output-scroll::-webkit-scrollbar { width: 4px; }
  .output-scroll::-webkit-scrollbar-track { background: var(--bg-secondary); }
  .output-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── 5. Provider definitions ────────────────────────────────────────────────────
PROVIDERS = {
    "ollama": {
        "label":    "Ollama (Local)",
        "icon":     "🖥️",
        "css":      "provider-ollama",
        "needs_key": False,
        "key_label": None,
        "key_env":   None,
        "models": [
            # ── General Purpose ──────────────────────────────────────────
            "gemma3:12b",
            "gemma3:4b",
            
            
        ],
        "default_model": "gemma3:12b",
        "note": "Runs locally. No API key needed. Start with: ollama serve",
    },
    "ollama_cloud": {
        "label":    "Ollama Cloud",
        "icon":     "☁️",
        "css":      "provider-ollama",
        "needs_key": True,
        "key_label": "Ollama Account Token",
        "key_env":   "OLLAMA_API_KEY",
        "models": [
            # ── Cloud models — no download needed, add :cloud tag ────────
            "gpt-oss:20b-cloud",
            "qwen3.5:cloud"
        ],
        "default_model": "deepseek-r1:cloud",
        "note": "No download needed. Sign in at ollama.com and get your token.",
    },
    "groq": {
        "label":    "Groq",
        "icon":     "⚡",
        "css":      "provider-groq",
        "needs_key": True,
        "key_label": "Groq API Key",
        "key_env":   "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile", 
            "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it",
        ],
        "default_model": "llama-3.3-70b-versatile",
        "note": "Free tier available. Get key at console.groq.com",
    },
    
}

SCENARIOS = {
    "1 · Dark Mode Setup": {"query": "How do I enable dark mode in my account?", "customer_id": "CUST-003"},
    "2 · API Access (Starter Plan)": {"query": "I'm on the Starter plan, but I need to integrate with your API for my automation workflow. What are my options?", "customer_id": "CUST-001"},
    "3 · API Rate Limit Contradiction": {"query": "Your documentation says the Pro plan includes unlimited API calls, but I'm seeing rate limit errors after 1000 calls/month. I've checked my account and it shows Pro. Is this a bug?", "customer_id": "CUST-002"},
    "4 · SLA Violation (Critical)": {"query": "I've been waiting for support response for 10 days on a critical production issue. My company has a contract with a 24-hour SLA guarantee. This is now costing us $500/day in lost revenue. Please verify if the SLA was violated and escalate this immediately.", "customer_id": "CUST-003"},
    "5 · Seat License Overage": {"query": "Our company just migrated from the competitor platform. We have 15 users, but the plan shows only 10 seats. Can you help me understand the licensing model and figure out how to set up all our users?", "customer_id": "CUST-001"},
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

# ── 6. LLM configuration helpers ──────────────────────────────────────────────
def apply_llm_config(provider: str, model: str, api_key: str = ""):
    """Write the chosen LLM config to environment variables so base_agent picks it up."""
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"]    = model

    p = PROVIDERS[provider]
    if p["needs_key"] and api_key:
        os.environ[p["key_env"]] = api_key

    # Also update config.yaml so CLI runs reflect the choice
    try:
        import yaml
        cfg_path = Path("config.yaml")
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        cfg.setdefault("llm", {})
        cfg["llm"]["provider"] = provider
        cfg["llm"]["model"]    = model
        cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))
    except Exception:
        pass

    # Reset LLM string cache in base_agent
    try:
        import agents.base_agent as ba
        ba._CONFIG = None
    except Exception:
        pass

    st.session_state.llm_provider   = provider
    st.session_state.llm_model      = model
    st.session_state.llm_configured = True
    logger.info(f"[UI] LLM set to {provider}/{model}")


def get_api_key_from_env(provider: str) -> str:
    """Read existing API key from environment if already set."""
    env_key = PROVIDERS[provider].get("key_env")
    return os.getenv(env_key, "") if env_key else ""


# ── 7. Background thread ───────────────────────────────────────────────────────
def _run_query_thread(query: str, customer_id: str, session_id: str, q: Queue):
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


# ── 8. Render helpers ──────────────────────────────────────────────────────────
def render_header():
    provider = st.session_state.llm_provider
    model    = st.session_state.llm_model
    p        = PROVIDERS.get(provider, PROVIDERS["ollama"])
    st.markdown(f"""
    <div class="app-header">
      <div style="font-size:2.2rem;">🤖</div>
      <div style="flex:1">
        <div class="app-header-title">TechCorp Support Intelligence</div>
        <div class="app-header-sub">CrewAI · Langfuse Cloud · 5 Specialized Agents</div>
      </div>
      <div>
        <span class="provider-pill {p['css']}">{p['icon']} {p['label']}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                     color:var(--text-muted);display:block;text-align:right;margin-top:4px">
          {model}
        </span>
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


def render_llm_selector():
    """Full LLM provider + model selector rendered in the sidebar."""
    st.markdown("### 🤖 LLM Provider")

    provider_labels = {k: f"{v['icon']} {v['label']}" for k, v in PROVIDERS.items()}
    current_provider = st.session_state.llm_provider

    # Provider radio buttons
    selected_provider = st.radio(
        "Provider",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: provider_labels[k],
        index=list(PROVIDERS.keys()).index(current_provider),
        label_visibility="collapsed",
    )

    p = PROVIDERS[selected_provider]

    # Model selector
    current_model = (
        st.session_state.llm_model
        if st.session_state.llm_provider == selected_provider
        else p["default_model"]
    )
    model_index = (
        p["models"].index(current_model)
        if current_model in p["models"]
        else 0
    )
    selected_model = st.selectbox(
        "Model",
        options=p["models"],
        index=model_index,
    )

    # API key input (shown only for non-local providers)
    api_key = ""
    if p["needs_key"]:
        existing_key = get_api_key_from_env(selected_provider)
        masked = f"{'*' * 20}{existing_key[-4:]}" if len(existing_key) > 4 else ""
        api_key = st.text_input(
            p["key_label"],
            value="",
            type="password",
            placeholder=masked or f"Enter your {p['key_label']}...",
        )
        if not api_key and existing_key:
            api_key = existing_key  # use existing env key if not re-entered
        st.markdown(
            f'<div style="font-size:0.68rem;color:var(--text-muted);margin-top:-8px">{p["note"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        icon = "☁️" if selected_provider == "ollama_cloud" else "✅"
        color = "var(--accent)" if selected_provider == "ollama_cloud" else "var(--accent-green)"
        st.markdown(
            f'<div style="font-size:0.68rem;color:{color};margin-top:4px">{icon} {p["note"]}</div>',
            unsafe_allow_html=True,
        )

    # Warning for cloud models without tool calling
    if selected_provider == "ollama_cloud":
        st.markdown("""
        <div style="background:rgba(240,136,62,0.1);border:1px solid rgba(240,136,62,0.3);
                    border-radius:8px;padding:0.6rem 0.8rem;margin:0.4rem 0;
                    font-size:0.72rem;color:#f0883e;font-family:'JetBrains Mono',monospace;">
          ⚠️ Requires <b>ollama signin</b> in terminal first.<br>
          Only qwen3 and llama3.3 variants support tool calling.
        </div>
        """, unsafe_allow_html=True)

    # Apply button
    config_changed = (
        selected_provider != st.session_state.llm_provider or
        selected_model    != st.session_state.llm_model
    )

    btn_label = "✅ Applied" if not config_changed else "⚡ Apply Configuration"
    if st.button(btn_label, disabled=not config_changed, use_container_width=True):
        apply_llm_config(selected_provider, selected_model, api_key)
        st.success(f"Switched to {p['label']} / {selected_model}")
        st.rerun()

    # Current config badge
    curr_p = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])
    st.markdown(
        f'<div style="margin-top:8px"><span class="provider-pill {curr_p["css"]}">'
        f'{curr_p["icon"]} {st.session_state.llm_provider} · {st.session_state.llm_model}'
        f'</span></div>',
        unsafe_allow_html=True,
    )


def render_sidebar_details():
    result = st.session_state.current_result
    if not result:
        return

    st.markdown('<div class="card-title">Last Run — Agents Invoked</div>', unsafe_allow_html=True)
    active = result.get("agents_used", [])
    chips = "".join(
        f'<span class="agent-chip {"active" if k in active else ""}">{m["icon"]} {m["label"]}</span>'
        for k, m in AGENT_META.items()
    )
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


# ── 9. Main ────────────────────────────────────────────────────────────────────
def main():
    # Apply any saved LLM config to env vars on each rerun
    if st.session_state.llm_configured:
        os.environ["LLM_PROVIDER"] = st.session_state.llm_provider
        os.environ["LLM_MODEL"]    = st.session_state.llm_model
        p = PROVIDERS.get(st.session_state.llm_provider, {})
        if p.get("key_env") and st.session_state.llm_api_key:
            os.environ[p["key_env"]] = st.session_state.llm_api_key

    render_header()
    render_metrics()
    st.markdown("<hr>", unsafe_allow_html=True)

    main_col, side_col = st.columns([3, 1], gap="large")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with side_col:
        render_llm_selector()

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("### 🧪 Test Scenarios")
        for name, data in SCENARIOS.items():
            if st.button(name, key=f"btn_{name}"):
                st.session_state.pending_query    = data["query"]
                st.session_state.pending_customer = data["customer_id"]
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

        customer_label = st.selectbox(
            "Customer",
            options=CUSTOMER_KEYS,
            index=st.session_state.customer_index,
        )
        customer_id = CUSTOMER_OPTIONS[customer_label]

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
            query      = query_input.strip()
            session_id = st.session_state.session_id

            st.session_state.running = True
            st.session_state.chat_history.append({"role": "user", "content": query})

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

            provider = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])

            if error:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": (
                        f"[ERROR] Investigation failed:\n{error}\n\n"
                        f"Provider: {provider['label']} · Model: {st.session_state.llm_model}\n\n"
                        "Please check:\n"
                        "• Ollama is running if using local model  (ollama serve)\n"
                        "• API key is set correctly for cloud providers\n"
                        "• Model name is valid for the selected provider\n"
                        "• .env file has Langfuse keys"
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
                try:
                    rp = Path("results/query_results.json")
                    existing = json.loads(rp.read_text()).get("results", []) if rp.exists() else []
                    existing.append({
                        "session_id":     result["session_id"],
                        "timestamp":      datetime.now(timezone.utc).isoformat(),
                        "query":          query,
                        "customer_id":    customer_id,
                        "llm_provider":   st.session_state.llm_provider,
                        "llm_model":      st.session_state.llm_model,
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
        lf = "🟢 Langfuse" if is_langfuse_enabled() else "🔴 Langfuse"
    except Exception:
        lf = "🔴 Langfuse"

    curr_p = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])
    st.markdown(f"""
    <div style="position:fixed;bottom:0;left:0;right:0;background:var(--bg-secondary);
                border-top:1px solid var(--border);padding:6px 20px;
                display:flex;justify-content:space-between;align-items:center;
                font-family:'JetBrains Mono',monospace;font-size:0.68rem;z-index:999;">
      <span style="color:var(--text-muted)">TechCorp Support AI</span>
      <span style="color:var(--text-muted)">
        {curr_p['icon']} {st.session_state.llm_provider} · {st.session_state.llm_model}
        &nbsp;|&nbsp; Session: {st.session_state.session_id}
        &nbsp;|&nbsp; {lf}
      </span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
