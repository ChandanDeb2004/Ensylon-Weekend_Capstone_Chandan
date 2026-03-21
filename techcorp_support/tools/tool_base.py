"""
tool_base.py
Shared base class for all mock tools.
Handles: realistic latency, random failures, incomplete data, Langfuse logging.
"""

import time
import random
import logging
import functools
from typing import Any, Callable, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def with_tool_behavior(tool_name: str, failure_rate: float = 0.03):
    """
    Decorator that wraps any tool function with:
      - Simulated latency (100–800ms)
      - Random failure injection
      - Structured logging for Langfuse
      - Graceful error return instead of crash
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            start_time = time.time()
            call_id = f"{tool_name}_{int(start_time * 1000)}"

            # Simulated latency
            latency_ms = random.randint(100, 800)
            time.sleep(latency_ms / 1000)

            # Random failure injection
            if random.random() < failure_rate:
                error_type = random.choice(["DatabaseTimeout", "NetworkError", "ServiceUnavailable"])
                elapsed = time.time() - start_time
                logger.warning(
                    f"[TOOL FAILURE] {tool_name} | call_id={call_id} "
                    f"| error={error_type} | latency={elapsed:.3f}s"
                )
                return {
                    "success": False,
                    "tool": tool_name,
                    "call_id": call_id,
                    "error": error_type,
                    "error_message": f"Simulated {error_type}: tool temporarily unavailable.",
                    "data": None,
                    "latency_ms": int(elapsed * 1000),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Execute actual tool logic
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(
                    f"[TOOL SUCCESS] {tool_name} | call_id={call_id} | latency={elapsed:.3f}s"
                )
                return {
                    "success": True,
                    "tool": tool_name,
                    "call_id": call_id,
                    "data": result,
                    "latency_ms": int(elapsed * 1000),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:
                elapsed = time.time() - start_time
                logger.error(
                    f"[TOOL ERROR] {tool_name} | call_id={call_id} "
                    f"| exception={str(exc)} | latency={elapsed:.3f}s"
                )
                return {
                    "success": False,
                    "tool": tool_name,
                    "call_id": call_id,
                    "error": "UnexpectedError",
                    "error_message": str(exc),
                    "data": None,
                    "latency_ms": int(elapsed * 1000),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        return wrapper
    return decorator


def format_tool_result(result: dict) -> str:
    """
    Convert a tool result dict into a clean string for the LLM to consume.
    Handles both success and failure states.
    """
    if not result["success"]:
        return (
            f"[TOOL CALL FAILED]\n"
            f"Tool: {result['tool']}\n"
            f"Error Type: {result.get('error', 'Unknown')}\n"
            f"Message: {result.get('error_message', 'No details available.')}\n"
            f"Instruction: Handle this gracefully — use partial data if available, "
            f"or note the gap in your findings."
        )

    data = result["data"]
    if data is None:
        return f"[TOOL RETURNED NO DATA]\nTool: {result['tool']}\nNote this as incomplete information."

    return (
        f"[TOOL SUCCESS]\n"
        f"Tool: {result['tool']}\n"
        f"Retrieved at: {result['timestamp']}\n"
        f"Data:\n{_format_data(data)}"
    )


def _format_data(data: Any) -> str:
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    elif isinstance(data, list):
        return "\n".join(f"  - {item}" for item in data)
    else:
        return f"  {data}"
