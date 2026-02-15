"""Bridge for using chunked documents as a risk knowledge base.

Uses the doc_chunker from multi-agent-investment-committee to
process risk-related PDFs (prospectuses, risk factor filings,
research notes on risk management) into a structured KB.

The chunks can be:
    1. Fed to an LLM for risk-aware analysis
    2. Used to validate risk model assumptions
    3. Stored as reference material for post-trade review

Connects your PDF chunking work (with page_range support) to
the backtest-lab risk module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def chunk_risk_document(
    pdf_path: str | Path,
    page_range: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Chunk a risk-related PDF into a knowledge base.

    Uses the doc_chunker from multi-agent-investment-committee
    with the page_range support you added.

    Args:
        pdf_path: Path to the PDF.
        page_range: Optional pages to extract (e.g., "23-41").
        output_path: Optional path to save chunks as JSON.

    Returns:
        List of chunk dicts from process_uploads().
    """
    from bridges.committee_bridge import load_doc_chunker

    process_uploads, format_kb_for_prompt = load_doc_chunker()

    if process_uploads is None:
        logger.error("doc_chunker not available")
        return []

    results = process_uploads([str(pdf_path)], page_range=page_range)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved %d chunks to %s", len(results), output_path)

    return results


def format_risk_kb(
    chunks: list[dict[str, Any]],
    max_tokens: int = 12_000,
) -> str:
    """Format chunked risk documents for LLM prompt injection.

    Args:
        chunks: Output from chunk_risk_document().
        max_tokens: Token budget.

    Returns:
        Formatted string ready for prompt injection.
    """
    from bridges.committee_bridge import load_doc_chunker

    _, format_kb_for_prompt = load_doc_chunker()

    if format_kb_for_prompt is None:
        logger.error("format_kb_for_prompt not available")
        return ""

    return format_kb_for_prompt(chunks, max_tokens=max_tokens)


def build_risk_kb_from_directory(
    directory: str | Path,
    page_ranges: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Chunk all PDFs in a directory into a risk KB.

    Args:
        directory: Directory containing risk-related PDFs.
        page_ranges: Optional {filename: page_range} overrides.

    Returns:
        Combined list of chunk dicts from all PDFs.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.error("Not a directory: %s", directory)
        return []

    all_chunks: list[dict[str, Any]] = []
    pdfs = sorted(dir_path.glob("*.pdf"))

    for pdf in pdfs:
        pr = None
        if page_ranges and pdf.name in page_ranges:
            pr = page_ranges[pdf.name]

        chunks = chunk_risk_document(pdf, page_range=pr)
        all_chunks.extend(chunks)
        logger.info("Chunked %s: %d docs", pdf.name, len(chunks))

    return all_chunks
