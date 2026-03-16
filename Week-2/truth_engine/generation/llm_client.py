"""
generation/llm_client.py

Multi-provider LLM client with automatic fallback.

Provider hierarchy (primary → fallback):
    1. Gemini  (google-generativeai) — primary
    2. Groq    (groq)                — fallback

Each provider is a concrete subclass of :class:`BaseLLMClient`, which
owns all retry logic, backoff, token tracking, and error handling.
Subclasses implement only ``_call_once`` and ``_is_retryable``.

The :class:`FallbackClient` orchestrates the provider chain — if the
primary provider exhausts all retries, it hands off to the next provider
transparently.  The caller always receives a plain string and never
needs to know which provider served the response.

The :class:`LLMCallable` protocol remains the injectable interface
contract used across the rest of the codebase.
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Gemini defaults
GEMINI_DEFAULT_MODEL: str = "gemini-2.5-flash"
GEMINI_DEFAULT_MAX_TOKENS: int = 1024

# Groq defaults
GROQ_DEFAULT_MODEL: str = "llama-3.1-8b-instant"
GROQ_DEFAULT_MAX_TOKENS: int = 1024

# Shared retry configuration
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BASE_DELAY: float = 1.0    # seconds before first retry
DEFAULT_MAX_DELAY: float = 30.0    # cap on backoff delay
JITTER_RANGE: float = 0.5          # ± seconds of random jitter


# ---------------------------------------------------------------------------
# Token usage tracking  
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Accumulated token usage across all API calls in a session.

    Attributes:
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens produced.
        total_calls: Total number of successful API calls.
        failed_calls: Total number of calls that failed after all retries.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0
    failed_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def record_success(self, input_tokens: int, output_tokens: int) -> None:
        """Record token counts from a successful API call.

        Args:
            input_tokens: Input token count from the API response.
            output_tokens: Output token count from the API response.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_calls += 1

    def record_failure(self) -> None:
        """Increment the failed call counter."""
        self.failed_calls += 1

    def __str__(self) -> str:
        return (
            f"TokenUsage(input={self.input_tokens}, "
            f"output={self.output_tokens}, "
            f"total={self.total_tokens}, "
            f"calls={self.total_calls}, "
            f"failed={self.failed_calls})"
        )


# ---------------------------------------------------------------------------
# Protocol — injectable callable contract  (unchanged from original)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMCallable(Protocol):
    """Protocol defining the callable interface for all LLM clients.

    Any object implementing ``__call__(prompt, system_prompt) -> str``
    satisfies this protocol and can be injected wherever an LLM callable
    is expected (e.g. :class:`~resolution.ConflictDetector`).
    """

    def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Send a prompt and return the LLM's text response.

        Args:
            prompt: The user-turn prompt string.
            system_prompt: Optional system instruction string.

        Returns:
            The LLM's text response as a plain string.
        """
        ...


# ---------------------------------------------------------------------------
# Shared retry helper  
# ---------------------------------------------------------------------------

def _compute_backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute exponential backoff delay with random jitter.

    Formula: ``min(base_delay * 2^attempt, max_delay) ± jitter``

    Args:
        attempt: Zero-based attempt index (0 = first retry).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.

    Returns:
        Delay in seconds to wait before the next retry attempt.
    """
    exponential = base_delay * (2 ** attempt)
    jitter = random.uniform(-JITTER_RANGE, JITTER_RANGE)  # noqa: S311
    return min(max(0.0, exponential + jitter), max_delay)


# ---------------------------------------------------------------------------
# Abstract base — owns all retry / logging / token-tracking logic
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract base class for all LLM provider clients.

    Owns the retry loop, exponential backoff, token usage accumulation,
    and graceful error degradation.  Subclasses implement only two
    methods:

    - :meth:`_call_once` — make a single raw API call and return a
      ``(text, input_tokens, output_tokens)`` tuple.
    - :meth:`_is_retryable` — classify an exception as retryable or not.

    This design ensures retry logic is written exactly once and every
    provider benefits from identical reliability behaviour.

    Attributes:
        provider_name: Human-readable provider identifier for logging.
        model: Model identifier string in use.
        usage: :class:`TokenUsage` instance for this client.
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        max_tokens: int,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> None:
        """Initialise shared retry and tracking state.

        Args:
            provider_name: Display name used in log messages.
            model: Model identifier string.
            max_tokens: Maximum completion tokens to request.
            max_retries: Retry attempts before giving up (excludes the
                initial attempt).
            base_delay: Base backoff delay in seconds.
            max_delay: Hard cap on backoff delay in seconds.
        """
        self.provider_name = provider_name
        self.model = model
        self.usage: TokenUsage = TokenUsage()
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    @abstractmethod
    def _call_once(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> tuple[str, int, int]:
        """Make a single raw API call to the provider.

        Args:
            prompt: User-turn prompt string.
            system_prompt: Optional system instruction string.

        Returns:
            A 3-tuple of ``(response_text, input_tokens, output_tokens)``.

        Raises:
            Any provider-specific API exception on failure.
        """
        ...

    @abstractmethod
    def _is_retryable(self, exc: Exception) -> bool:
        """Classify an exception as retryable or not.

        Args:
            exc: The caught exception.

        Returns:
            ``True`` if the error is transient and a retry may succeed.
        """
        ...

    def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Send a prompt with retry logic, backoff, and token logging.

        Args:
            prompt: The user-turn prompt string.
            system_prompt: Optional system instruction string.

        Returns:
            The LLM's text response as a plain string, or a
            ``"[LLM ERROR] ..."`` string if all retries are exhausted.
            Never raises — callers can check for the ``[LLM ERROR]``
            prefix or inspect ``self.usage.failed_calls``.
        """
        if not prompt.strip():
            logger.warning(
                "[%s] __call__ received an empty prompt. Returning empty.",
                self.provider_name,
            )
            return ""

        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = _compute_backoff(
                    attempt - 1, self._base_delay, self._max_delay
                )
                logger.warning(
                    "[%s] Retrying (attempt %d/%d) after %.2fs. Last error: %s",
                    self.provider_name, attempt, self._max_retries,
                    delay, last_exception,
                )
                time.sleep(delay)

            try:
                logger.debug(
                    "[%s] Call attempt %d: model='%s', prompt_chars=%d.",
                    self.provider_name, attempt + 1, self.model, len(prompt),
                )

                text, input_tokens, output_tokens = self._call_once(
                    prompt, system_prompt
                )

                self.usage.record_success(input_tokens, output_tokens)
                logger.info(
                    "[%s] Call succeeded: input_tokens=%d, output_tokens=%d, "
                    "session_total=%d.",
                    self.provider_name, input_tokens, output_tokens,
                    self.usage.total_tokens,
                )
                return text

            except Exception as exc:  # noqa: BLE001
                last_exception = exc

                if not self._is_retryable(exc):
                    logger.error(
                        "[%s] Non-retryable error (attempt %d): %s",
                        self.provider_name, attempt + 1, exc,
                    )
                    break

                if attempt == self._max_retries:
                    logger.error(
                        "[%s] Failed after %d attempt(s). Last error: %s",
                        self.provider_name, self._max_retries + 1, exc,
                    )
                    break

                logger.warning(
                    "[%s] Retryable error (attempt %d/%d): %s",
                    self.provider_name, attempt + 1,
                    self._max_retries + 1, exc,
                )

        self.usage.record_failure()
        error_message = (
            f"[LLM ERROR] [{self.provider_name}] Could not be reached after "
            f"{self._max_retries + 1} attempt(s). Last error: {last_exception}"
        )
        logger.error(error_message)
        return error_message


# ---------------------------------------------------------------------------
# Gemini client  (primary provider)
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    """Gemini provider client using the ``google-generativeai`` SDK.

    Primary provider in the :class:`FallbackClient` chain.

    The system prompt is injected via Gemini's ``system_instruction``
    parameter at model initialisation time.  Because ``system_instruction``
    is set on the ``GenerativeModel`` object, a new model instance is
    created per call when a system prompt is provided.  Calls without a
    system prompt reuse a shared instance for efficiency.

    Attributes:
        model: Gemini model identifier string.

    Example:
        >>> client = GeminiClient()
        >>> response = client("What is BM25?")
    """

    def __init__(
        self,
        model: str = GEMINI_DEFAULT_MODEL,
        max_tokens: int = GEMINI_DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        api_key: str | None = None,
    ) -> None:
        """Initialise the Gemini client.

        Args:
            model: Gemini model identifier.
            max_tokens: Maximum output tokens.
            max_retries: Retry attempts on transient errors.
            base_delay: Base backoff delay in seconds.
            max_delay: Hard cap on backoff delay.
            api_key: Google AI API key.  If ``None``, reads from the
                ``GOOGLE_API_KEY`` environment variable.
        """
        super().__init__(
            provider_name="Gemini",
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )

        import google.generativeai as genai

        if api_key:
            genai.configure(api_key=api_key)
        # If api_key is None, genai reads GOOGLE_API_KEY from environment.

        self._genai = genai
        # Shared model instance for calls without a system prompt.
        self._default_model = genai.GenerativeModel(
            model_name=self.model,
            generation_config={"max_output_tokens": self._max_tokens},
        )

        logger.info(
            "GeminiClient initialised: model='%s', max_tokens=%d, "
            "max_retries=%d.",
            model, max_tokens, max_retries,
        )

    def _call_once(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> tuple[str, int, int]:
        """Make a single Gemini API call.

        Args:
            prompt: User-turn prompt string.
            system_prompt: Optional system instruction.  Injected via
                ``system_instruction`` on a fresh ``GenerativeModel``
                instance when provided.

        Returns:
            ``(response_text, input_tokens, output_tokens)``
        """
        if system_prompt:
            model = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
                generation_config={"max_output_tokens": self._max_tokens},
            )
        else:
            model = self._default_model

        response = model.generate_content(prompt)

        text = response.text
        # Gemini token counts live in usage_metadata.
        input_tokens: int = getattr(
            response.usage_metadata, "prompt_token_count", 0
        ) or 0
        output_tokens: int = getattr(
            response.usage_metadata, "candidates_token_count", 0
        ) or 0

        return text, input_tokens, output_tokens

    def _is_retryable(self, exc: Exception) -> bool:
        """Classify a Gemini exception as retryable.

        Args:
            exc: The caught exception.

        Returns:
            ``True`` for rate limit and server-side errors.
        """
        try:
            from google.api_core import exceptions as google_exc

            if isinstance(exc, google_exc.ResourceExhausted):
                return True   # 429 rate limit
            if isinstance(exc, google_exc.ServiceUnavailable):
                return True   # 503
            if isinstance(exc, google_exc.InternalServerError):
                return True   # 500
            if isinstance(exc, google_exc.DeadlineExceeded):
                return True   # timeout
        except ImportError:
            pass

        # Fallback: retry on any non-specified exception type.
        # Better to over-retry than to silently drop a recoverable error.
        return True


# ---------------------------------------------------------------------------
# Groq client  (fallback provider)
# ---------------------------------------------------------------------------

class GroqClient(BaseLLMClient):
    """Groq provider client using the ``groq`` SDK.

    Fallback provider in the :class:`FallbackClient` chain.  Groq's SDK
    mirrors the OpenAI SDK interface, making its exception hierarchy
    predictable and easy to classify.

    Attributes:
        model: Groq model identifier string.

    Example:
        >>> client = GroqClient()
        >>> response = client("What is BM25?")
    """

    def __init__(
        self,
        model: str = GROQ_DEFAULT_MODEL,
        max_tokens: int = GROQ_DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        api_key: str | None = None,
    ) -> None:
        """Initialise the Groq client.

        Args:
            model: Groq model identifier.
            max_tokens: Maximum output tokens.
            max_retries: Retry attempts on transient errors.
            base_delay: Base backoff delay in seconds.
            max_delay: Hard cap on backoff delay.
            api_key: Groq API key.  If ``None``, reads from the
                ``GROQ_API_KEY`` environment variable.
        """
        super().__init__(
            provider_name="Groq",
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )

        import groq as groq_sdk

        self._groq = groq_sdk.Groq(api_key=api_key)

        logger.info(
            "GroqClient initialised: model='%s', max_tokens=%d, "
            "max_retries=%d.",
            model, max_tokens, max_retries,
        )

    def _call_once(
        self,
        prompt: str,
        system_prompt: str | None,
    ) -> tuple[str, int, int]:
        """Make a single Groq API call.

        Groq uses an OpenAI-compatible messages format.  The system
        prompt is passed as a ``{"role": "system", ...}`` message
        prepended to the messages list.

        Args:
            prompt: User-turn prompt string.
            system_prompt: Optional system instruction string.

        Returns:
            ``(response_text, input_tokens, output_tokens)``
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._groq.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
        )

        text = response.choices[0].message.content or ""
        input_tokens: int = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens: int = getattr(response.usage, "completion_tokens", 0) or 0

        return text, input_tokens, output_tokens

    def _is_retryable(self, exc: Exception) -> bool:
        """Classify a Groq exception as retryable.

        Groq's exception hierarchy mirrors the OpenAI SDK.

        Args:
            exc: The caught exception.

        Returns:
            ``True`` for rate limit, server, and connection errors.
        """
        try:
            import groq as groq_sdk

            if isinstance(exc, groq_sdk.RateLimitError):
                return True
            if isinstance(exc, groq_sdk.APIConnectionError):
                return True
            if isinstance(exc, groq_sdk.APITimeoutError):
                return True
            if isinstance(exc, groq_sdk.APIStatusError):
                retryable_codes = {500, 502, 503, 504}
                return exc.status_code in retryable_codes
        except ImportError:
            pass

        return False


# ---------------------------------------------------------------------------
# Fallback orchestrator
# ---------------------------------------------------------------------------

class FallbackClient:
    """Multi-provider LLM client with automatic failover.

    Tries each provider in the configured chain in order.  If a provider
    returns an ``[LLM ERROR]`` response (all retries exhausted), the next
    provider in the chain is tried.  The first successful response is
    returned.

    If all providers fail, a final graceful error string is returned.

    The fallback decision is encapsulated here — no other module needs
    to know that multiple providers exist.

    Satisfies the :class:`LLMCallable` protocol.

    Attributes:
        usage: Aggregated :class:`TokenUsage` across all providers that
            were called in the session.

    Example:
        >>> client = FallbackClient()   # Gemini → Groq
        >>> response = client(
        ...     prompt="What is BM25?",
        ...     system_prompt="You are a concise technical assistant.",
        ... )
        >>> print(client.usage)
    """

    def __init__(
        self,
        providers: list[BaseLLMClient] | None = None,
    ) -> None:
        """Initialise the fallback client with a provider chain.

        Args:
            providers: Ordered list of :class:`BaseLLMClient` instances.
                The first provider is tried first; subsequent providers
                are fallbacks.  Defaults to ``[GeminiClient(), GroqClient()]``
                when ``None``.
        """
        if providers is None:
            providers = [GeminiClient(), GroqClient()]

        if not providers:
            raise ValueError(
                "FallbackClient requires at least one provider."
            )

        self._providers = providers
        provider_names = [p.provider_name for p in providers]
        logger.info(
            "FallbackClient initialised with provider chain: %s",
            " → ".join(provider_names),
        )

    @property
    def usage(self) -> TokenUsage:
        """Aggregated token usage across all providers.

        Returns:
            A new :class:`TokenUsage` instance summing all providers'
            usage counters.
        """
        agg = TokenUsage()
        for provider in self._providers:
            agg.input_tokens  += provider.usage.input_tokens
            agg.output_tokens += provider.usage.output_tokens
            agg.total_calls   += provider.usage.total_calls
            agg.failed_calls  += provider.usage.failed_calls
        return agg

    def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Send a prompt, falling back through providers as needed.

        Args:
            prompt: The user-turn prompt string.
            system_prompt: Optional system instruction string.

        Returns:
            The first successful text response from any provider in the
            chain.  Returns a ``"[LLM ERROR] ..."`` string if all
            providers fail.  Never raises.
        """
        for provider in self._providers:
            logger.info(
                "FallbackClient: trying provider '%s'.", provider.provider_name
            )
            response = provider(prompt=prompt, system_prompt=system_prompt)

            if not response.startswith("[LLM ERROR]"):
                if provider != self._providers[0]:
                    # Log which fallback provider served the request.
                    logger.warning(
                        "FallbackClient: response served by fallback "
                        "provider '%s'.",
                        provider.provider_name,
                    )
                return response

            logger.warning(
                "FallbackClient: provider '%s' failed — trying next provider.",
                provider.provider_name,
            )

        all_names = ", ".join(p.provider_name for p in self._providers)
        final_error = (
            f"[LLM ERROR] All providers failed ({all_names}). "
            f"No response could be generated."
        )
        logger.error(final_error)
        return final_error