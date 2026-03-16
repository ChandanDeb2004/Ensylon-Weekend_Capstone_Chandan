# PDF + table parsing
"""
ingestion/pdf_ingestor.py

Handles ingestion of PDF documents into a uniform list of text chunks.
Each chunk represents one page of the PDF. Tables are extracted and
rendered as Markdown to preserve their structure for downstream LLM use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pdfplumber
from pdfplumber.page import Page

# ---------------------------------------------------------------------------
# Module-level logger — callers can attach handlers as needed.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_cell(cell: Optional[str]) -> str:
    """Coerce a table cell value to a clean string.

    ``pdfplumber`` may return ``None`` for empty cells and may embed
    newlines inside a single cell (e.g. wrapped header text).  Both cases
    produce broken Markdown if left unhandled.

    Args:
        cell: Raw cell value returned by ``pdfplumber``.

    Returns:
        A single-line string safe for embedding in a Markdown table cell.
    """
    if cell is None:
        return ""
    # Collapse internal newlines / extra whitespace within one cell.
    return " ".join(str(cell).split())


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """Convert a 2-D cell grid from ``pdfplumber`` into a Markdown table.

    The first row is treated as the header row and a separator line is
    inserted beneath it, conforming to GitHub-Flavoured Markdown (GFM)
    table syntax that most LLMs are trained on.

    Args:
        table: A list of rows, where each row is a list of cell strings
            (or ``None`` for empty cells), as returned by
            ``pdfplumber``'s ``page.extract_tables()``.

    Returns:
        A multi-line string containing the rendered Markdown table, or an
        empty string if the table contains fewer than two rows (header +
        at least one data row) or zero columns.

    Example:
        >>> raw = [["Step", "Action", "Time"], ["1", "Restart", "5"]]
        >>> print(_table_to_markdown(raw))
        | Step | Action  | Time |
        |------|---------|------|
        | 1    | Restart | 5    |
    """
    if not table or len(table) < 1:
        return ""

    # Normalise every row to the same column count (handle ragged tables).
    max_cols: int = max(len(row) for row in table)
    if max_cols == 0:
        return ""

    sanitized: List[List[str]] = []
    for row in table:
        padded = [_sanitize_cell(cell) for cell in row]
        # Pad short rows so every row has the same number of columns.
        padded.extend([""] * (max_cols - len(padded)))
        sanitized.append(padded)

    def _build_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header_row: str = _build_row(sanitized[0])
    # Separator: at least three dashes per column for valid GFM.
    separator: str = "| " + " | ".join(["---"] * max_cols) + " |"

    data_rows: List[str] = [_build_row(row) for row in sanitized[1:]]

    lines: List[str] = [header_row, separator, *data_rows]
    return "\n".join(lines)


def _extract_page_content(page: Page) -> Optional[str]:
    """Extract all readable content from a single ``pdfplumber`` page.

    Strategy:
    1.  Extract every table on the page and render each as a Markdown block.
    2.  Extract plain text from the page (``pdfplumber`` excludes table
        bounding boxes from ``extract_text`` by default when tables are
        detected, reducing duplication).
    3.  Merge plain text blocks and Markdown tables into a single string
        in document order (text first, tables appended).

    If neither text nor tables yield any content the page is considered
    empty (likely a scanned image) and ``None`` is returned.

    Args:
        page: A ``pdfplumber.Page`` object for the page being processed.

    Returns:
        A non-empty string containing the page's content with tables
        rendered as Markdown, or ``None`` if the page is empty/image-only.
    """
    parts: List[str] = []

    # ------------------------------------------------------------------ #
    # 1. Extract structured tables.
    # ------------------------------------------------------------------ #
    try:
        tables: List[List[List[Optional[str]]]] = page.extract_tables()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Table extraction failed on page %d: %s", page.page_number, exc)
        tables = []

    markdown_tables: List[str] = []
    for table in tables:
        md = _table_to_markdown(table)
        if md:
            markdown_tables.append(md)

    # ------------------------------------------------------------------ #
    # 2. Extract plain text (pdfplumber strips table regions automatically
    #    when tables are present, minimising duplication).
    # ------------------------------------------------------------------ #
    try:
        raw_text: Optional[str] = page.extract_text()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Text extraction failed on page %d: %s", page.page_number, exc)
        raw_text = None

    cleaned_text: str = (raw_text or "").strip()
    if cleaned_text:
        parts.append(cleaned_text)

    # Append Markdown tables after plain text.
    parts.extend(markdown_tables)

    if not parts:
        return None

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(file_path: str | Path) -> List[str]:
    """Ingest a PDF file and return its content as a list of page chunks.

    Each element of the returned list corresponds to one non-empty page.
    Tables detected on a page are rendered as GitHub-Flavoured Markdown
    tables so that downstream LLM components can parse their structure.
    Pages that contain no extractable content (e.g. scanned images or
    rotated text rendered as a bitmap) are skipped with a warning log.

    Args:
        file_path: Absolute or relative path to the PDF file.  Accepts
            either a plain ``str`` or a :class:`pathlib.Path` object.

    Returns:
        A list of strings, one per non-empty page, where table content
        is represented as Markdown.  Returns an empty list if the PDF
        contains no extractable content.

    Raises:
        FileNotFoundError: If ``file_path`` does not point to an existing
            file.
        ValueError: If ``file_path`` does not have a ``.pdf`` extension.
        RuntimeError: If ``pdfplumber`` cannot open the file (e.g. the
            file is password-protected or corrupt).

    Example:
        >>> chunks = ingest_pdf("documents/manual.pdf")
        >>> for i, chunk in enumerate(chunks, start=1):
        ...     print(f"--- Page {i} ---")
        ...     print(chunk)
    """
    path = Path(file_path)

    # ------------------------------------------------------------------ #
    # Pre-flight validation — fail fast with actionable messages.
    # ------------------------------------------------------------------ #
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: '{path}'")

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got '{path.suffix}' for path: '{path}'"
        )

    logger.info("Starting PDF ingestion: %s", path)

    chunks: List[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            total_pages: int = len(pdf.pages)
            logger.info("PDF has %d page(s).", total_pages)

            for page in pdf.pages:
                page_number: int = page.page_number  # 1-based in pdfplumber
                logger.debug("Processing page %d / %d …", page_number, total_pages)

                content = _extract_page_content(page)

                if content is None:
                    # Scanned / image-only page — log and skip gracefully.
                    logger.warning(
                        "Page %d of '%s' yielded no extractable content "
                        "(possible scanned image or non-standard encoding). "
                        "Skipping page.",
                        page_number,
                        path.name,
                    )
                    continue

                chunks.append(content)

    except pdfplumber.PDFSyntaxError as exc:
        raise RuntimeError(
            f"pdfplumber could not parse '{path}'. "
            "The file may be password-protected, corrupt, or not a valid PDF."
        ) from exc

    logger.info(
        "Ingestion complete: %d / %d page(s) extracted from '%s'.",
        len(chunks),
        total_pages if "total_pages" in dir() else 0,
        path.name,
    )
    return chunks