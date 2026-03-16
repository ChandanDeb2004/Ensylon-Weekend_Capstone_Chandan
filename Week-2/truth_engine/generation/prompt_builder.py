# Builds final prompt with context + citations
"""
generation/prompt_builder.py

Constructs structured prompts for the final LLM generation call.

The prompt architecture has four sections:
  1. System role   — persona, citation rules, behaviour constraints.
  2. Context blocks — retrieved evidence, labeled by source and tier.
  3. Conflict notice — explicit disclosure of any source overrides.
  4. Final instruction — uncertainty escape hatch ("say I don't know").

Citation strings are generated programmatically from chunk metadata
so the LLM never needs to invent source names or page numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_ROLE_TEMPLATE: str = """\
You are a precise technical assistant. Your answers must be:
- Grounded exclusively in the context passages provided below.
- Cited using the source identifiers given in each passage header.
- Honest about uncertainty: if the context does not contain enough
  information to answer confidently, you must say "I don't know based
  on the provided sources" rather than speculating.

Citation format: When you use information from a passage, cite it
inline using its identifier, e.g. [SRC-1] or [SRC-2].
Never cite a source you did not directly use.\
"""

CONTEXT_BLOCK_HEADER_TEMPLATE: str = (
    "[{src_id}] {citation_string}\n"
    "{separator}\n"
    "{text}"
)

CONFLICT_NOTICE_TEMPLATE: str = """\
⚠ SOURCE CONFLICT NOTICE
The following source conflicts were detected and resolved before this
context was assembled. The overriding source takes precedence:

{entries}

Do not attempt to reconcile the overridden content. Rely only on the
winning source for the conflicting facts.\
"""

CONFLICT_ENTRY_TEMPLATE: str = (
    "• '{winner_source}' (Tier {winner_tier}) overrides "
    "'{loser_source}' (Tier {loser_tier}): {loser_snippet}"
)

FINAL_INSTRUCTION_TEMPLATE: str = """\
Using only the context passages above, answer the following question.
Cite every factual claim with its [SRC-N] identifier.
If the context is insufficient, respond with:
"I don't know based on the provided sources."

Question: {query}\
"""

SEPARATOR: str = "-" * 60
SNIPPET_MAX_CHARS: int = 120


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class BuiltPrompt:
    """The assembled prompt ready for the LLM client.

    Attributes:
        system_prompt: The system role instruction string.  Should be
            passed as the ``system`` parameter in the API call, not
            prepended to ``user_prompt``.
        user_prompt: The full user-turn prompt containing context blocks,
            conflict notice, and the final question.
        citations: List of pre-formatted citation strings, one per
            context block.  Included for downstream use in the final
            answer post-processor.
        source_map: Dict mapping ``"SRC-N"`` identifiers to their
            corresponding chunk metadata dicts for traceability.
    """

    system_prompt: str
    user_prompt: str
    citations: list[str] = field(default_factory=list)
    source_map: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_citation_string(metadata: dict[str, Any]) -> str:
    """Build a human-readable citation string from chunk metadata.

    Args:
        metadata: A chunk's metadata dict.  Reads ``source_name``,
            ``page_number``, and ``section``.

    Returns:
        A formatted citation string, e.g.::

            "According to manual_v2.pdf (Page 14, §Restart Procedures)"
            "According to incidents.csv"
    """
    source_name: str = metadata.get("source_name", "Unknown Source")
    page: int | None = metadata.get("page_number")
    section: str = metadata.get("section", "").strip()

    parts: list[str] = [f"According to {source_name}"]

    location_parts: list[str] = []
    if page is not None and str(page) != "0":
        location_parts.append(f"Page {page}")
    if section:
        location_parts.append(f"§{section}")

    if location_parts:
        parts.append(f"({', '.join(location_parts)})")

    return " ".join(parts)


def _truncate(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Truncate text to a maximum character length with an ellipsis.

    Args:
        text: Input string.
        max_chars: Maximum character count before truncation.

    Returns:
        The original string if within ``max_chars``, otherwise the
        string truncated to ``max_chars - 3`` characters with ``"..."``
        appended.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_context_blocks(
    chunks: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    """Build formatted context block strings from a list of chunks.

    Args:
        chunks: List of chunk dicts, each with ``"text"`` and
            ``"metadata"`` keys.

    Returns:
        A 3-tuple of:
        - List of formatted context block strings.
        - List of citation strings (one per chunk).
        - Source map: ``{"SRC-1": metadata_dict, ...}``.
    """
    blocks: list[str] = []
    citations: list[str] = []
    source_map: dict[str, dict[str, Any]] = {}

    for idx, chunk in enumerate(chunks, start=1):
        src_id = f"SRC-{idx}"
        metadata = chunk.get("metadata", {})
        citation_string = _build_citation_string(metadata)
        tier = metadata.get("source_tier", "?")
        confidence = chunk.get("confidence", 1.0)

        # Header line includes tier and confidence for full transparency.
        full_header = (
            f"{citation_string}  |  "
            f"Tier {tier}  |  "
            f"Confidence {confidence:.0%}"
        )

        block = CONTEXT_BLOCK_HEADER_TEMPLATE.format(
            src_id=src_id,
            citation_string=full_header,
            separator=SEPARATOR,
            text=chunk.get("text", "").strip(),
        )
        blocks.append(block)
        citations.append(f"[{src_id}] {citation_string}")
        source_map[src_id] = metadata

    return blocks, citations, source_map


def _build_conflict_notice(
    suppressed_chunks: list[dict[str, Any]],
) -> str | None:
    """Build the conflict notice section from suppressed chunk records.

    Args:
        suppressed_chunks: List of annotated suppressed chunk dicts
            as returned by :class:`~resolution.SourcePrioritizer`.

    Returns:
        A formatted conflict notice string, or ``None`` if
        ``suppressed_chunks`` is empty.
    """
    if not suppressed_chunks:
        return None

    entries: list[str] = []
    for chunk in suppressed_chunks:
        entry = CONFLICT_ENTRY_TEMPLATE.format(
            winner_source=chunk.get("overridden_by_source", "unknown"),
            winner_tier=chunk.get("overridden_by_tier", "?"),
            loser_source=chunk.get("original_source", "unknown"),
            loser_tier=chunk.get("original_tier", "?"),
            loser_snippet=_truncate(chunk.get("text", "")),
        )
        entries.append(entry)

    return CONFLICT_NOTICE_TEMPLATE.format(entries="\n".join(entries))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Assembles structured four-section prompts for LLM generation.

    Produces a :class:`BuiltPrompt` dataclass containing separate system
    and user prompts, pre-generated citation strings, and a source map
    for downstream traceability.

    Attributes:
        system_role_template: The system prompt template in use.

    Example:
        >>> builder = PromptBuilder()
        >>> built = builder.build(
        ...     query="How long does Procedure X take?",
        ...     final_chunks=result.final_chunks,
        ...     suppressed_chunks=result.suppressed_chunks,
        ... )
        >>> print(built.user_prompt)
    """

    def __init__(
        self,
        system_role_template: str = SYSTEM_ROLE_TEMPLATE,
    ) -> None:
        """Initialise the builder with an optional custom system template.

        Args:
            system_role_template: Override the default system role
                instruction.  The default template is appropriate for
                most technical Q&A use cases.
        """
        self.system_role_template = system_role_template

    def build(
        self,
        query: str,
        final_chunks: list[dict[str, Any]],
        suppressed_chunks: list[dict[str, Any]] | None = None,
    ) -> BuiltPrompt:
        """Construct the full four-section prompt for an LLM generation call.

        Args:
            query: The user's original query string.
            final_chunks: Surviving chunks from
                :class:`~resolution.TruthResolver`, ordered by
                relevance (most relevant first).
            suppressed_chunks: Optional list of annotated suppressed
                chunk dicts from :class:`~resolution.SourcePrioritizer`.
                When provided, a conflict notice section is included.
                Pass ``None`` or ``[]`` when no conflicts were detected.

        Returns:
            A :class:`BuiltPrompt` dataclass instance.

        Raises:
            ValueError: If ``query`` is empty or ``final_chunks`` is
                empty.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not final_chunks:
            logger.warning(
                "build called with no final_chunks for query '%s'. "
                "Returning no-context prompt.",
                query[:60],
            )
            return BuiltPrompt(
                system_prompt=self.system_role_template,
                user_prompt=(
                    f"No relevant context passages were found for the query:\n\n"
                    f"{query.strip()}\n\n"
                    "Respond with: "
                    "\"I don't know based on the provided sources.\""
                ),
                citations=[],
                source_map={},
            )

        suppressed = suppressed_chunks or []
        logger.info(
            "Building prompt: %d context chunk(s), %d suppressed, "
            "query='%s …'",
            len(final_chunks),
            len(suppressed),
            query[:60],
        )

        # ---------------------------------------------------------------- #
        # Section 1: System role (returned separately, not in user_prompt).
        # ---------------------------------------------------------------- #
        system_prompt = self.system_role_template

        # ---------------------------------------------------------------- #
        # Section 2: Context blocks.
        # ---------------------------------------------------------------- #
        context_blocks, citations, source_map = _build_context_blocks(
            final_chunks
        )
        context_section = "\n\n".join(context_blocks)

        # ---------------------------------------------------------------- #
        # Section 3: Conflict notice (conditional).
        # ---------------------------------------------------------------- #
        conflict_notice = _build_conflict_notice(suppressed)

        # ---------------------------------------------------------------- #
        # Section 4: Final instruction with the user query.
        # ---------------------------------------------------------------- #
        final_instruction = FINAL_INSTRUCTION_TEMPLATE.format(
            query=query.strip()
        )

        # ---------------------------------------------------------------- #
        # Assemble user prompt — conflict notice is only included when
        # conflicts exist, keeping the prompt clean for the common case.
        # ---------------------------------------------------------------- #
        user_sections: list[str] = ["CONTEXT PASSAGES:", context_section]

        if conflict_notice:
            user_sections.append(conflict_notice)

        user_sections.append(final_instruction)
        user_prompt = "\n\n".join(user_sections)

        logger.debug(
            "Prompt built: system=%d chars, user=%d chars, "
            "%d citation(s).",
            len(system_prompt),
            len(user_prompt),
            len(citations),
        )

        return BuiltPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            citations=citations,
            source_map=source_map,
        )