"""
base_agent.py — Multi-provider LLM support
Supports: Ollama (local), Groq, Google Gemini, OpenAI
Switch provider in config.yaml or via LLM_PROVIDER env var.
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
                    "provider":    "ollama",
                    "model":       os.getenv("OLLAMA_MODEL", "gemma3:12b"),
                    "base_url":    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    "temperature": 0.1,
                }
            }
    return _CONFIG


def build_llm_string() -> str:
    """
    Returns a LiteLLM model string based on the configured provider.

    LiteLLM string formats:
      Ollama:  "ollama/gemma3:12b"
      Groq:    "groq/llama-3.1-70b-versatile"
      Gemini:  "gemini/gemini-1.5-pro"
      OpenAI:  "gpt-4o"

    Provider is read from config.yaml llm.provider field,
    or overridden by LLM_PROVIDER environment variable.
    """
    cfg      = load_config().get("llm", {})
    provider = os.getenv("LLM_PROVIDER", cfg.get("provider", "ollama")).lower()
    model    = os.getenv("LLM_MODEL",    cfg.get("model",    "gemma3:12b"))
    temp     = cfg.get("temperature", 0.1)

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", cfg.get("base_url", "http://localhost:11434"))
        os.environ["OLLAMA_API_BASE"] = base_url
        llm_str = f"ollama/{model}"

    elif provider == "ollama_cloud":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        os.environ["OLLAMA_API_BASE"] = base_url
        # Ensure model has -cloud suffix (not :cloud)
        
        llm_str = f"ollama/{model}"
        logger.info(
            f"[LLM] Ollama Cloud mode: model={model} | "
            f"Make sure you ran: ollama signin"
        )

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        os.environ["GROQ_API_KEY"] = api_key
        # Groq strict schema requirements
        os.environ["LITELLM_DROP_PARAMS"] = "True"
        if model in ("gemma3:12b", "gemma3:4b"):
            logger.warning(
                "[LLM] Groq does not support Gemma3 models. "
                "Switching to llama-3.3-70b-versatile. "
                "Update LLM_MODEL in your .env to suppress this warning."
            )
            model = "llama-3.3-70b-versatile"
        llm_str = f"groq/{model}"

    elif provider in ("gemini", "google"):
        api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        os.environ["GEMINI_API_KEY"] = api_key
        llm_str = f"gemini/{model}"

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        llm_str = model  # OpenAI uses model name directly e.g. "gpt-4o"

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        # Valid Anthropic model names — guard against typos
        VALID_ANTHROPIC = {
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        }
        if model not in VALID_ANTHROPIC:
            logger.warning(
                f"[LLM] Unknown Anthropic model '{model}'. "
                f"Switching to claude-3-5-haiku-20241022. "
                f"Valid models: {sorted(VALID_ANTHROPIC)}"
            )
            model = "claude-3-5-haiku-20241022"
        llm_str = f"anthropic/{model}"

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Choose from: ollama, groq, gemini, openai, anthropic"
        )

    logger.info(f"[LLM] Provider: {provider} | Model: {llm_str}")
    return llm_str



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
        
    )
    logger.info(f"[AGENT] Built: {role}")
    return agent
