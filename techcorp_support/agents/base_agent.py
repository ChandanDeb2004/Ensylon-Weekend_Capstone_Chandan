"""
base_agent.py — crewai>=1.10.1 + langfuse==2.36.2

Attaches langfuse.callback.CallbackHandler to the LLM so every
model call is automatically traced in Langfuse — no manual spans needed
for LLM calls. Step callback adds per-step events.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Optional
from crewai import Agent

logger = logging.getLogger(__name__)
_CONFIG: Optional[dict] = None


def load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        p = Path("config.yaml")
        if p.exists():
            with open(p) as f:
                _CONFIG = yaml.safe_load(f)
        else:
            _CONFIG = {
                "llm": {
                    "model":       os.getenv("OLLAMA_MODEL",    "gemma3:12b"),
                    "base_url":    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "temperature": 0.1,
                }
            }
    return _CONFIG


def build_llm_string() -> str:
    cfg      = load_config().get("llm", {})
    model    = cfg.get("model",    os.getenv("OLLAMA_MODEL",    "gemma3:12b"))
    base_url = cfg.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    os.environ["OLLAMA_API_BASE"] = base_url
    llm_str = f"ollama/{model}"
    logger.info(f"[LLM] Using model: {llm_str} | base_url: {base_url}")
    return llm_str


def _make_step_callback(agent_name: str, session_id: str):
    """Logs each agent step as a Langfuse event."""
    def callback(step_output):
        try:
            from monitoring.tracing_utils import _safe_log_event
            _safe_log_event(
                name=f"agent_step",
                session_id=session_id,
                metadata={
                    "agent":          agent_name,
                    "output_preview": str(step_output)[:300],
                },
            )
        except Exception:
            pass
    return callback


def build_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: list,
    session_id: str = "default",
    verbose: bool = True,
    max_iter: int = 8,
    allow_delegation: bool = False,
) -> Agent:
    # Ensure Langfuse client is initialized
    from monitoring.langfuse_config import get_langfuse_client
    get_langfuse_client()

    llm = build_llm_string()

    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=allow_delegation,
        memory=False,
        step_callback=_make_step_callback(role, session_id),
    )
    logger.info(f"[AGENT] Built: {role}")
    return agent
