"""
generation/
~~~~~~~~~~~

Station 5 of the RAG pipeline: prompt construction and LLM generation.

Two components working in sequence:

:class:`PromptBuilder`
    Assembles the four-section structured prompt from resolved chunks.
    Generates citation strings programmatically from chunk metadata so
    the LLM never needs to invent source names or page numbers.
    Returns a :class:`BuiltPrompt` dataclass consumed by the LLM client.

:class:`AnthropicClient`
    Reliability wrapper around the Anthropic Messages API.  Handles
    retry logic (exponential backoff with jitter), structured error
    handling, and per-call token logging with session accumulation.
    Satisfies the :class:`LLMCallable` protocol for clean injection.

Supporting types:

:class:`BuiltPrompt`
    Dataclass holding the assembled system prompt, user prompt,
    citation list, and source map.

:class:`LLMCallable`
    ``runtime_checkable`` Protocol defining the injectable LLM callable
    interface.  Any ``(prompt, system_prompt) -> str`` callable qualifies.

:class:`TokenUsage`
    Dataclass tracking input/output token counts and call statistics
    across a session.

Typical usage::

    from generation import PromptBuilder, AnthropicClient, BuiltPrompt

    builder = PromptBuilder()
    client  = AnthropicClient()

    built: BuiltPrompt = builder.build(
        query=query,
        final_chunks=result.final_chunks,
        suppressed_chunks=result.suppressed_chunks,
    )

    answer = client(
        prompt=built.user_prompt,
        system_prompt=built.system_prompt,
    )
"""

from generation.llm_client import  LLMCallable, TokenUsage
from generation.prompt_builder import BuiltPrompt, PromptBuilder

from generation.llm_client import (
    FallbackClient,    # add
    GeminiClient,      # add
    GroqClient,        # add
    LLMCallable,
    TokenUsage,
)

__all__ = [
    "PromptBuilder",
    "BuiltPrompt",
    "FallbackClient",  # add
    "GeminiClient",    # add
    "GroqClient",      # add
    "LLMCallable",
    "TokenUsage",
]
