# JSON/CSV support log parsing
"""
ingestion/json_ingestor.py

Handles ingestion of structured log data from JSON and CSV sources.
Each record is rendered into a human-readable sentence using a
caller-supplied template, making the data semantically accessible
to downstream LLM components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default template — callers should override this for their specific schema.
# Column names are referenced as {column_name} placeholders.
# ---------------------------------------------------------------------------
DEFAULT_TEMPLATE: str = (
    "Problem: {problem_description}. "
    "Fix Applied: {fix}. "
    "Outcome: {outcome}."
)


def _render_record(record: dict, template: str, row_index: int) -> str | None:
    """Render a single data record into a human-readable sentence.

    Uses :func:`str.format_map` to interpolate record field values into
    the template string.  Missing keys are caught per-record so that one
    malformed row does not abort the entire ingestion run.

    Args:
        record: A dictionary mapping column names to their string values
            for a single row.
        template: A Python format string whose placeholders correspond to
            column names in ``record``.  Example::

                "Problem: {problem_description}. Fix: {fix}."

        row_index: The zero-based integer index of this row in the source
            file, used exclusively for diagnostic log messages.

    Returns:
        The rendered sentence string, or ``None`` if a required placeholder
        key is absent from ``record``.
    """
    try:
        return template.format_map(record)
    except KeyError as exc:
        logger.warning(
            "Row %d is missing column %s required by the template. "
            "Skipping row.",
            row_index,
            exc,
        )
        return None


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a JSON or CSV file into a :class:`pandas.DataFrame`.

    Dispatch is based solely on the file extension.  Both paths normalise
    all column values to strings and replace ``NaN`` with empty strings so
    that template interpolation never encounters float ``nan`` values.

    Args:
        path: A validated :class:`pathlib.Path` pointing to a ``.json``
            or ``.csv`` file.

    Returns:
        A :class:`pandas.DataFrame` with all values coerced to ``str``.

    Raises:
        ValueError: If the file extension is not ``.json`` or ``.csv``.
        RuntimeError: If ``pandas`` fails to parse the file content.
    """
    ext = path.suffix.lower()
    try:
        if ext == ".json" or ext == ".jsonl":
            # ``orient="records"`` is the most common export format for
            # log systems; ``lines=True`` handles newline-delimited JSON.
            # We try standard first, then fall back to lines format.
            try:
                df = pd.read_json(path)
            except ValueError:
                logger.debug(
                    "Standard JSON parse failed for '%s'; "
                    "retrying as newline-delimited JSON.",
                    path.name,
                )
                df = pd.read_json(path, lines=True)

        elif ext == ".csv":
            df = pd.read_csv(path)

        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                "json_ingestor accepts '.json' and '.csv' only."
            )
    except ValueError:
        # Re-raise ValueError (unsupported extension) as-is.
        raise
    except Exception as exc:
        raise RuntimeError(
            f"pandas failed to parse '{path}'. "
            "Verify the file is well-formed."
        ) from exc

    # Coerce all values to str; replace NaN with empty string.
    return df.fillna("").astype(str)


def ingest_json(
    file_path: str | Path,
    template: str = DEFAULT_TEMPLATE,
) -> List[str]:
    """Ingest a JSON or CSV file and return a list of human-readable chunks.

    Each row in the source file is rendered into a natural-language sentence
    by substituting column values into ``template``.  Rows for which the
    template cannot be rendered (missing columns) are skipped with a warning.

    The same code path handles both ``.json`` and ``.csv`` files via
    :mod:`pandas`, making the function format-agnostic from the caller's
    perspective.

    Args:
        file_path: Path to the source ``.json`` or ``.csv`` file.
        template: A Python :func:`str.format_map`-compatible string whose
            placeholders match column names in the source file.  Defaults
            to :data:`DEFAULT_TEMPLATE`.  Callers *should* supply a schema-
            specific template for best results::

                template = (
                    "Problem: {problem_description}. "
                    "Fix Applied: {fix}. "
                    "Outcome: {outcome}."
                )

    Returns:
        A list of rendered sentence strings, one per successfully processed
        row.  Returns an empty list if the file contains no rows or all
        rows fail to render.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the file extension is not ``.json`` or ``.csv``.
        RuntimeError: If the file cannot be parsed by ``pandas``.

    Example:
        >>> chunks = ingest_json(
        ...     "data/incident_log.csv",
        ...     template="Problem: {problem_description}. Fix: {fix}.",
        ... )
        >>> print(chunks[0])
        Problem: daemon crash on startup. Fix: restart daemon service.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: '{path}'")

    logger.info("Starting JSON/CSV ingestion: %s", path)

    df = _load_dataframe(path)

    if df.empty:
        logger.warning("'%s' contains no rows. Returning empty chunk list.", path.name)
        return []

    logger.info("Loaded %d row(s) from '%s'.", len(df), path.name)

    chunks: List[str] = []
    for index, row in df.iterrows():
        rendered = _render_record(row.to_dict(), template, int(str(index)))
        if rendered is not None:
            chunks.append(rendered)

    logger.info(
        "Ingestion complete: %d / %d row(s) rendered from '%s'.",
        len(chunks),
        len(df),
        path.name,
    )
    return chunks