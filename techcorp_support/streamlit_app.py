"""
streamlit_app.py — TechCorp Support Intelligence
Fixed: render_chat now calls render_ai_response for rich output.
Fixed: humanize_response removed — synthesis already writes human prose.
"""

import os, re
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
        "llm_provider":      "ollama",
        "llm_model":         "gemma3:12b",
        "llm_api_key":       "",
        "llm_configured":    True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

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

st.set_page_config(
    page_title="TechCorp Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

  /* User message bubble */
  .chat-user {
    background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px 10px 2px 10px; padding: 0.9rem 1.1rem; margin: 0.6rem 0;
    color: var(--text-primary) !important;
  }
  .chat-role { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; font-family: 'JetBrains Mono', monospace; }
  .chat-role-user { color: #58a6ff !important; }
  .chat-role-ai   { color: #3fb950 !important; }

  /* AI response card */
  .ai-response-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 2px 10px 10px 10px; padding: 1.2rem 1.4rem; margin: 0.6rem 0;
  }
  .ai-response-prose {
    font-size: 0.95rem; line-height: 1.75; color: var(--text-primary) !important;
    margin-bottom: 0.8rem;
  }

  /* Status banners */
  .status-resolved {
    background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.35);
    padding: 0.55rem 0.9rem; border-radius: 8px; margin-bottom: 0.8rem;
    font-weight: 600; font-size: 0.82rem; color: #3fb950;
    display: flex; align-items: center; gap: 6px;
  }
  .status-escalated {
    background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.4);
    padding: 0.55rem 0.9rem; border-radius: 8px; margin-bottom: 0.8rem;
    font-weight: 600; font-size: 0.82rem; color: #f85149;
    display: flex; align-items: center; gap: 6px;
  }

  /* Technical details expander */
  .find-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; font-family: 'JetBrains Mono', monospace;
    padding: 0.4rem 0; margin-bottom: 0.3rem; margin-top: 0.6rem;
  }
  .find-label-account  { color: #58a6ff; border-bottom: 1px solid rgba(88,166,255,0.2);  padding-bottom: 4px; }
  .find-label-feature  { color: #3fb950; border-bottom: 1px solid rgba(63,185,80,0.2);   padding-bottom: 4px; }
  .find-label-contract { color: #f0883e; border-bottom: 1px solid rgba(240,136,62,0.2);  padding-bottom: 4px; }
  .find-label-esc      { color: #f85149; border-bottom: 1px solid rgba(248,81,73,0.2);   padding-bottom: 4px; }
  .find-key   { font-size: 0.78rem; color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace; }
  .find-val   { font-size: 0.82rem; color: var(--text-primary) !important; }
  .find-val-g { color: #3fb950 !important; }
  .find-val-r { color: #f85149 !important; }
  .find-val-y { color: #f0883e !important; }

  .conflict-box {
    background: rgba(240,136,62,0.08); border: 1px solid rgba(240,136,62,0.3);
    border-radius: 8px; padding: 0.7rem 0.9rem; margin: 0.4rem 0;
    font-size: 0.78rem; color: #f0883e !important; font-family: 'JetBrains Mono', monospace;
  }

  /* Progress */
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem; }
  .card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted) !important; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace; }
  .progress-step { display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary) !important; border-bottom: 1px solid var(--bg-hover); }
  .progress-step:last-child { border-bottom: none; color: var(--accent-green) !important; }

  .agent-chip { display: inline-flex; align-items: center; gap: 0.4rem; background: var(--bg-hover); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 0.73rem; color: var(--text-secondary) !important; margin: 3px 2px; font-family: 'JetBrains Mono', monospace; }
  .agent-chip.active { border-color: var(--accent); color: var(--accent) !important; background: rgba(88,166,255,0.08); }
  .escalation-alert { background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.4); border-radius: 10px; padding: 1rem 1.2rem; margin: 0.6rem 0; }
  .escalation-alert-title { font-size: 0.9rem; font-weight: 700; color: #f85149 !important; margin-bottom: 0.4rem; }

  .provider-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; border: 1px solid; margin: 2px; }
  .provider-ollama       { background: rgba(63,185,80,0.12);  color: #3fb950; border-color: rgba(63,185,80,0.3); }
  .provider-ollama_cloud { background: rgba(63,185,80,0.12);  color: #3fb950; border-color: rgba(63,185,80,0.3); }
  .provider-groq         { background: rgba(248,81,73,0.12);  color: #f85149; border-color: rgba(248,81,73,0.3); }
  .provider-gemini       { background: rgba(88,166,255,0.12); color: #58a6ff; border-color: rgba(88,166,255,0.3); }
  .provider-openai       { background: rgba(188,140,255,0.12);color: #bc8cff; border-color: rgba(188,140,255,0.3); }
  .provider-anthropic    { background: rgba(240,136,62,0.12); color: #f0883e; border-color: rgba(240,136,62,0.3); }

  .stButton > button { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; transition: all 0.2s !important; width: 100% !important; }
  .stButton > button:hover { border-color: var(--accent) !important; background: rgba(88,166,255,0.08) !important; color: var(--accent) !important; }
  .stTextArea textarea, .stTextInput input { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; border-radius: 8px !important; font-size: 0.9rem !important; }
  .stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important; }
  .stSelectbox > div > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; border-radius: 8px !important; }
  hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
  .output-scroll { max-height: 520px; overflow-y: auto; padding-right: 4px; }
  .output-scroll::-webkit-scrollbar { width: 4px; }
  .output-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
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
            "qwen3.5:cloud",
            "deepseek-v3.1:671b-cloud"
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
    "1 · Dark Mode Setup":           {"query":"How do I enable dark mode in my account?","customer_id":"CUST-003"},
    "2 · API Access (Starter Plan)": {"query":"I'm on the Starter plan, but I need to integrate with your API. What are my options?","customer_id":"CUST-001"},
    "3 · API Rate Limit Conflict":   {"query":"Your docs say Pro has unlimited API calls, but I'm hitting limits at 1000/month. My account shows Pro. Is this a bug?","customer_id":"CUST-002"},
    "4 · SLA Violation 🔴":          {"query":"I've been waiting 10 days on a critical production issue. My contract has a 24-hour SLA guarantee. This is costing us $500/day. Please verify the SLA violation and escalate immediately.","customer_id":"CUST-003"},
    "5 · Seat License Overage":      {"query":"We migrated from a competitor and have 15 users, but the plan shows only 10 seats. Help me understand the licensing.","customer_id":"CUST-001"},
}

CUSTOMER_OPTIONS = {
    "CUST-001 · Acme Corp (Starter)":     "CUST-001",
    "CUST-002 · Globex Inc (Pro)":         "CUST-002",
    "CUST-003 · Initech LLC (Enterprise)": "CUST-003",
    "CUST-004 · Umbrella Ltd (Suspended)": "CUST-004",
}
CUSTOMER_KEYS = list(CUSTOMER_OPTIONS.keys())
AGENT_META = {
    "account":    {"icon":"👤","label":"Account"},
    "feature":    {"icon":"⚙️","label":"Feature"},
    "contract":   {"icon":"📄","label":"Contract"},
    "escalation": {"icon":"🚨","label":"Escalation"},
}

# ── Parsing helpers ────────────────────────────────────────────────────────────
def _parse_block(raw, name):
    m = re.search(rf"{name}:\s*\n(.*?)(?=\n[A-Z_]{{3,}}:|\Z)", raw, re.DOTALL|re.IGNORECASE)
    if not m: return []
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip().lstrip("*-•").strip()
        if ":" in line:
            k,_,v = line.partition(":")
            k,v = k.strip(), v.strip()
            if k and v and len(k)<70 and len(v)<500:
                rows.append((k,v))
    return rows

def _vc(v):
    vl = v.lower()
    if any(x in vl for x in ["yes","active","✅","none found","current","not needed","resolve_auto","high confidence","no overage"]): return "find-val-g"
    if any(x in vl for x in ["no","escalate","critical","breach","violated","suspended","❌","failed","error"]): return "find-val-r"
    if any(x in vl for x in ["medium","unknown","low","to be determined","unavailable"]): return "find-val-y"
    return "find-val"

# ── Status banner ──────────────────────────────────────────────────────────────
def render_status_banner(escalated: bool, ticket: str):
    if escalated:
        st.markdown(f'<div class="status-escalated">🚨 Escalated{f" — Ticket {ticket}" if ticket else ""}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-resolved">✅ Resolved automatically</div>', unsafe_allow_html=True)

# ── Technical details expander ─────────────────────────────────────────────────
def render_technical_dropdown(structured: str, conflicts: list):
    account    = _parse_block(structured, "ACCOUNT_FINDINGS")
    feature    = _parse_block(structured, "FEATURE_FINDINGS")
    contract   = _parse_block(structured, "CONTRACT_FINDINGS")
    escalation = _parse_block(structured, "ESCALATION_FINDINGS")
    structured_is_error = structured.strip().startswith("[CREW EXECUTION ERROR]")
    has_any = not structured_is_error and any([account, feature, contract, escalation, conflicts])
    if not has_any:
        return

    with st.expander("📋 Technical Details", expanded=False):
        if account:
            st.markdown('<div class="find-label find-label-account">👤 Account</div>', unsafe_allow_html=True)
            for k, v in account:
                c1, c2 = st.columns([2, 3])
                c1.markdown(f'<span class="find-key">{k}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span class="{_vc(v)}">{v}</span>', unsafe_allow_html=True)

        if feature:
            st.markdown('<div class="find-label find-label-feature">⚙️ Feature</div>', unsafe_allow_html=True)
            for k, v in feature:
                c1, c2 = st.columns([2, 3])
                c1.markdown(f'<span class="find-key">{k}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span class="{_vc(v)}">{v}</span>', unsafe_allow_html=True)

        if contract:
            st.markdown('<div class="find-label find-label-contract">📄 Contract</div>', unsafe_allow_html=True)
            for k, v in contract:
                c1, c2 = st.columns([2, 3])
                c1.markdown(f'<span class="find-key">{k}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span class="{_vc(v)}">{v}</span>', unsafe_allow_html=True)

        if conflicts:
            st.markdown('<div class="find-label" style="color:#f0883e;border-bottom:1px solid rgba(240,136,62,0.2)">⚠️ Conflicts</div>', unsafe_allow_html=True)
            for c in conflicts:
                st.markdown(f'<div class="conflict-box">{c}</div>', unsafe_allow_html=True)

        if escalation:
            st.markdown('<div class="find-label find-label-esc">🚨 Escalation</div>', unsafe_allow_html=True)
            for k, v in escalation:
                c1, c2 = st.columns([2, 3])
                c1.markdown(f'<span class="find-key">{k}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span class="{_vc(v)}">{v}</span>', unsafe_allow_html=True)

# ── AI response renderer ───────────────────────────────────────────────────────
def render_ai_response(result: dict):
    """
    Called inside st.chat_message('assistant').
    Shows: status banner → human prose → collapsible technical details.
    No second LLM call. No humanize_response(). The prose comes from
    the synthesize_response() LLM call already done in orchestrator.py.
    """
    raw       = result.get("final_response", "")
    escalated = result.get("escalated", False)
    ticket    = result.get("ticket_id", "")
    conflicts = result.get("conflicts", [])

    # Split prose from structured data
    if "---STRUCTURED_DATA---" in raw:
        prose, _, structured = raw.partition("---STRUCTURED_DATA---")
    else:
        prose, structured = raw, ""

    prose = prose.strip()
    # Guard against leaked separator or error in prose
    if "---STRUCTURED_DATA---" in prose:
        prose = prose.split("---STRUCTURED_DATA---")[0].strip()
    is_error = prose.startswith("[CREW EXECUTION ERROR]") or prose.startswith("Investigation failed")

    render_status_banner(escalated, ticket)

    if is_error:
        st.error(prose)
    elif prose:
        st.markdown(f'<div class="ai-response-prose">{prose}</div>', unsafe_allow_html=True)

    render_technical_dropdown(structured, conflicts)

# ── Chat render ────────────────────────────────────────────────────────────────
def render_chat():
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#484f58;">
          <div style="font-size:3rem;margin-bottom:1rem;">💬</div>
          <div style="font-size:1rem;font-weight:500;color:#8b949e;">Ask a support question to begin</div>
          <div style="font-size:0.78rem;margin-top:0.5rem;">Select a scenario from the sidebar or type below</div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown('<div class="output-scroll">', unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="chat-user">'
                f'<div class="chat-role chat-role-user">▸ You</div>'
                f'{turn["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # AI turn — use st.chat_message so render_ai_response widgets work correctly
            with st.chat_message("assistant"):
                st.markdown('<div class="chat-role chat-role-ai">◈ TechCorp AI</div>', unsafe_allow_html=True)
                if turn.get("result"):
                    render_ai_response(turn["result"])
                else:
                    # Plain text fallback (errors etc.)
                    st.write(turn["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# ── LLM config ─────────────────────────────────────────────────────────────────
def apply_llm_config(provider, model, api_key=""):
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"]    = model
    p = PROVIDERS[provider]
    if p["needs_key"] and api_key:
        os.environ[p["key_env"]] = api_key
    try:
        import yaml; cp = Path("config.yaml")
        cfg = yaml.safe_load(cp.read_text()) if cp.exists() else {}
        cfg.setdefault("llm",{}).update({"provider":provider,"model":model})
        cp.write_text(yaml.dump(cfg, default_flow_style=False))
    except: pass
    try:
        import agents.base_agent as ba; ba._CONFIG = None
    except: pass
    st.session_state.llm_provider   = provider
    st.session_state.llm_model      = model
    st.session_state.llm_configured = True

def get_api_key_from_env(provider):
    env_key = PROVIDERS[provider].get("key_env")
    return os.getenv(env_key, "") if env_key else ""

# ── Background thread ──────────────────────────────────────────────────────────
def _run_query_thread(query, customer_id, session_id, q):
    try:
        from agents.orchestrator import run_support_crew
        result = run_support_crew(query=query, customer_id=customer_id, session_id=session_id,
                                   progress_callback=lambda s: q.put(("progress", s)))
        q.put(("done", result))
    except Exception as e:
        q.put(("error", str(e)))

# ── Sidebar components ─────────────────────────────────────────────────────────
def render_header():
    p = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])
    st.markdown(f"""
    <div class="app-header">
      <div style="font-size:2.2rem;">🤖</div>
      <div style="flex:1">
        <div class="app-header-title">TechCorp Support Intelligence</div>
        <div class="app-header-sub">CrewAI · Langfuse Cloud · 5 Specialized Agents</div>
      </div>
      <div>
        <span class="provider-pill {p['css']}">{p['icon']} {p['label']}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--text-muted);display:block;text-align:right;margin-top:4px">
          {st.session_state.llm_model}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

def render_metrics():
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val in [
        (c1,"Total Queries",     st.session_state.total_queries),
        (c2,"Escalations",       st.session_state.total_escalations),
        (c3,"Conflicts Resolved",st.session_state.total_conflicts),
        (c4,"Session ID",        st.session_state.session_id),
    ]:
        col.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

def render_llm_selector():
    st.markdown("### 🤖 LLM Provider")
    current_provider = st.session_state.llm_provider
    selected_provider = st.radio("Provider",
        options=list(PROVIDERS.keys()),
        format_func=lambda k: f"{PROVIDERS[k]['icon']} {PROVIDERS[k]['label']}",
        index=list(PROVIDERS.keys()).index(current_provider),
        label_visibility="collapsed")
    p = PROVIDERS[selected_provider]
    cur_model = st.session_state.llm_model if st.session_state.llm_provider==selected_provider else p["default_model"]
    midx = p["models"].index(cur_model) if cur_model in p["models"] else 0
    selected_model = st.selectbox("Model", options=p["models"], index=midx)

    api_key = ""
    if p["needs_key"]:
        existing = get_api_key_from_env(selected_provider)
        api_key = st.text_input(p["key_label"], value="", type="password",
            placeholder=f"{'*'*20}{existing[-4:]}" if len(existing)>4 else f"Enter your {p['key_label']}...")
        if not api_key and existing: api_key = existing
        st.markdown(f'<div style="font-size:0.68rem;color:var(--text-muted);margin-top:-8px">{p["note"]}</div>', unsafe_allow_html=True)
    else:
        icon = "☁️" if selected_provider=="ollama_cloud" else "✅"
        color = "var(--accent)" if selected_provider=="ollama_cloud" else "var(--accent-green)"
        st.markdown(f'<div style="font-size:0.68rem;color:{color};margin-top:4px">{icon} {p["note"]}</div>', unsafe_allow_html=True)

    if selected_provider=="ollama_cloud":
        st.markdown('<div style="background:rgba(240,136,62,0.1);border:1px solid rgba(240,136,62,0.3);border-radius:8px;padding:0.6rem 0.8rem;margin:0.4rem 0;font-size:0.72rem;color:#f0883e">⚠️ Run <b>ollama signin</b> first. Only qwen3/llama3.3 support tool calling.</div>', unsafe_allow_html=True)

    changed = selected_provider!=st.session_state.llm_provider or selected_model!=st.session_state.llm_model
    if st.button("✅ Applied" if not changed else "⚡ Apply Configuration", disabled=not changed, use_container_width=True):
        apply_llm_config(selected_provider, selected_model, api_key)
        st.success(f"Switched to {p['label']} / {selected_model}")
        st.rerun()

    curr_p = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])
    st.markdown(f'<div style="margin-top:8px"><span class="provider-pill {curr_p["css"]}">{curr_p["icon"]} {st.session_state.llm_provider} · {st.session_state.llm_model}</span></div>', unsafe_allow_html=True)

def render_sidebar_details():
    result = st.session_state.current_result
    if not result: return
    st.markdown('<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;color:#484f58;font-family:JetBrains Mono,monospace;margin-bottom:0.5rem">Last Run — Agents</div>', unsafe_allow_html=True)
    active = result.get("agents_used", [])
    chips = "".join(f'<span class="agent-chip {"active" if k in active else ""}">{m["icon"]} {m["label"]}</span>' for k,m in AGENT_META.items())
    st.markdown(f'<div style="margin-bottom:0.8rem">{chips}</div>', unsafe_allow_html=True)
    dur = result.get("duration_s",0); esc = result.get("escalated",False)
    c1,c2 = st.columns(2)
    c1.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.1rem">{dur}s</div><div class="metric-label">Duration</div></div>', unsafe_allow_html=True)
    color="#f85149" if esc else "#3fb950"; label="YES" if esc else "NO"
    c2.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.1rem;color:{color}">{label}</div><div class="metric-label">Escalated</div></div>', unsafe_allow_html=True)
    if esc and result.get("ticket_id"):
        st.markdown(f'<div class="escalation-alert"><div class="escalation-alert-title">🚨 {result["ticket_id"]}</div></div>', unsafe_allow_html=True)
    with st.expander("🔬 State JSON", expanded=False):
        st.json(result.get("state_dict",{}))
    ao = result.get("agent_outputs",{})
    if ao:
        with st.expander("📋 Raw Agent Outputs", expanded=False):
            for key, out in ao.items():
                meta = AGENT_META.get(key,{"icon":"•","label":key})
                st.markdown(f"**{meta['icon']} {meta['label']}**")
                st.code(out[:2000], language="text")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
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
                        st.session_state.customer_index = i; break
                st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔬 Last Run Details")
        render_sidebar_details()

    with main_col:
        st.markdown("### 💬 Conversation")
        render_chat()
        st.markdown("<hr>", unsafe_allow_html=True)

        customer_label = st.selectbox("Customer", options=CUSTOMER_KEYS, index=st.session_state.customer_index)
        customer_id = CUSTOMER_OPTIONS[customer_label]

        pending_q = st.session_state.pending_query
        if pending_q: st.session_state.pending_query = ""

        query_input = st.text_area("Support Query", value=pending_q, height=90,
            placeholder="Type a support question, or click a scenario above...")

        send_col, clear_col = st.columns([4, 1])
        with send_col:
            send_btn = st.button("🚀  Run Investigation", disabled=st.session_state.running, type="primary", use_container_width=True)
        with clear_col:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_result = None
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.rerun()

        if send_btn and query_input.strip() and not st.session_state.running:
            query      = query_input.strip()
            session_id = st.session_state.session_id
            st.session_state.running = True
            st.session_state.chat_history.append({"role":"user","content":query,"result":None})

            q: Queue = Queue()
            threading.Thread(target=_run_query_thread, args=(query,customer_id,session_id,q), daemon=True).start()

            placeholder = st.empty(); steps=[]; result=None; error=None; t0=time.time()
            while True:
                try:
                    kind, payload = q.get(timeout=0.3)
                    if kind == "progress":
                        steps.append(payload)
                        html = "".join(f'<div class="progress-step"><span>{"✅" if i==len(steps)-1 else "⏳"}</span> {s}</div>' for i,s in enumerate(steps))
                        placeholder.markdown(f'<div class="card"><div class="card-title">🔄 Investigating</div>{html}</div>', unsafe_allow_html=True)
                    elif kind == "done": result=payload; break
                    elif kind == "error": error=payload; break
                except Empty:
                    placeholder.markdown(f'<div class="card"><div class="card-title">⏳ Running... ({time.time()-t0:.0f}s)</div></div>', unsafe_allow_html=True)
                    if not any(t.daemon and t.is_alive() for t in threading.enumerate()): break

            placeholder.empty(); st.session_state.running = False
            provider = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])

            if error:
                st.session_state.chat_history.append({"role":"assistant","content":f"[ERROR] {error}","result":None})
            elif result:
                st.session_state.current_result      = result
                st.session_state.total_queries      += 1
                st.session_state.total_escalations  += int(result.get("escalated",False))
                st.session_state.total_conflicts    += len(result.get("conflicts",[]))
                # Store full result object — render_ai_response reads it directly
                st.session_state.chat_history.append({"role":"assistant","content":result["final_response"],"result":result})
                try:
                    rp = Path("results/query_results.json")
                    ex = json.loads(rp.read_text()).get("results",[]) if rp.exists() else []
                    ex.append({"session_id":result["session_id"],"timestamp":datetime.now(timezone.utc).isoformat(),"query":query,"customer_id":customer_id,"llm_provider":st.session_state.llm_provider,"llm_model":st.session_state.llm_model,"final_response":result["final_response"],"agents_used":result["agents_used"],"duration_s":result["duration_s"],"escalated":result["escalated"],"ticket_id":result.get("ticket_id"),"conflicts":result["conflicts"]})
                    rp.write_text(json.dumps({"results":ex},indent=2,default=str))
                except Exception as e: logger.warning(f"[UI] Save failed: {e}")
            st.rerun()

    try:
        from monitoring.langfuse_config import is_langfuse_enabled
        lf = "🟢 Langfuse" if is_langfuse_enabled() else "🔴 Langfuse"
    except: lf = "🔴 Langfuse"
    curr_p = PROVIDERS.get(st.session_state.llm_provider, PROVIDERS["ollama"])
    st.markdown(f"""
    <div style="position:fixed;bottom:0;left:0;right:0;background:var(--bg-secondary);border-top:1px solid var(--border);padding:6px 20px;display:flex;justify-content:space-between;align-items:center;font-family:'JetBrains Mono',monospace;font-size:0.68rem;z-index:999;">
      <span style="color:var(--text-muted)">TechCorp Support AI</span>
      <span style="color:var(--text-muted)">{curr_p['icon']} {st.session_state.llm_provider} · {st.session_state.llm_model} &nbsp;|&nbsp; Session: {st.session_state.session_id} &nbsp;|&nbsp; {lf}</span>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()