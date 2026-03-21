"""
base_agent.py
Shared base class and LLM factory for all agents.
Configures Ollama (qwen2.5) with Langfuse callback integration.
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
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                _CONFIG = yaml.safe_load(f)
        else:
            _CONFIG = {
                "llm": {
                    "model": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
                    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "temperature": 0.1,
                }
            }
    return _CONFIG


def build_llm(session_id: str = "default"):
    cfg = load_config()["llm"]
    model = cfg.get("model", "gemma3:12b")

    llm = f"ollama/{model}"

    logger.info(f"[LLM] Using model: {llm}")
    return llm

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
    """
    Factory function that builds a configured CrewAI Agent with:
    - Ollama (qwen2.5) as the LLM
    - Langfuse callback tracing
    - Retry / max_iter settings
    """
    llm = build_llm(session_id=session_id)
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=allow_delegation,
        memory=False,  # We manage memory externally via StateManager
    )
    logger.info(f"[AGENT] Built agent: {role}")
    return agent
