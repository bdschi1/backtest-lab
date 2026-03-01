"""Bridge to AAH (Am AI Hallucinating?) — factual integrity gate.

AAH screens LLM output for hallucinations using multiple independent
strategies: cross-claim NLI contradiction, source grounding (HHEM),
self-consistency, web verification, market data grounding, and citation
verification.

This bridge lets backtest-lab screen committee memos and research text
for factual integrity before they influence trading signals. Complements
redflag_bridge (compliance gate) with a factual accuracy gate.

Gate pattern:
    LLM output → [redflag: compliance] → [aah: factual integrity] → signal
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_AAH_PATH = Path("/Users/bdsm4/code/bds_repos/aah")


def _ensure_import():
    # AAH uses src layout — add src/ to path
    src_path = str(_AAH_PATH / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def screen_text(
    text: str,
    source_documents: list[str] | None = None,
    fail_threshold: float = 0.6,
) -> dict:
    """Screen LLM output text for hallucinations.

    Args:
        text: LLM-generated text to screen (memo, analysis, etc.).
        source_documents: Optional source texts for grounding (e.g.,
            the data-gathering output that the LLM was supposed to
            summarize). Enables HHEM source grounding strategy.
        fail_threshold: Hallucination probability above which the
            screen fails. Default 0.6.

    Returns:
        Dict with:
            - passed: bool — whether the text passed the screen
            - hallucination_probability: float 0.0-1.0
            - total_claims: int
            - flagged_claims: list of dicts for claims at HIGH or CRITICAL risk
            - critical_failures: list of failure reason strings
            - strategies_used: list of strategy IDs that ran
    """
    _ensure_import()
    try:
        from aah import HallucinationScreener
        from aah.config import ScreenerSettings

        settings = ScreenerSettings(ensemble_fail_threshold=fail_threshold)
        screener = HallucinationScreener(settings=settings)
        report = screener.screen(
            response_text=text,
            source_documents=source_documents or [],
        )

        flagged = []
        for cr in report.claim_results:
            if cr.risk_level in ("high", "critical"):
                flagged.append({
                    "claim": cr.claim.text,
                    "claim_type": cr.claim.claim_type.value,
                    "score": cr.ensemble_score,
                    "risk_level": cr.risk_level,
                    "is_critical": cr.is_critical,
                    "critical_reason": cr.critical_reason or "",
                })

        return {
            "passed": report.passed,
            "hallucination_probability": report.hallucination_probability,
            "total_claims": report.total_claims,
            "flagged_claims": flagged,
            "critical_failures": report.critical_failures,
            "strategies_used": report.strategies_used,
        }
    except ImportError:
        logger.warning("aah not available — skipping hallucination screen")
        return {
            "passed": True,
            "hallucination_probability": 0.0,
            "total_claims": 0,
            "flagged_claims": [],
            "critical_failures": [],
            "strategies_used": [],
            "error": "aah package not found at expected path",
        }


def screen_committee_memo(
    memo_text: str,
    gather_data_output: str | None = None,
    fail_threshold: float = 0.6,
) -> dict:
    """Screen a committee memo with optional source grounding.

    Convenience wrapper that passes the data-gathering phase output
    as source documents, enabling HHEM NLI grounding to verify the
    memo's claims against the data the LLM actually received.

    Args:
        memo_text: The PM synthesis memo text.
        gather_data_output: Raw text from the data-gathering phase.
            If provided, enables source grounding strategy.
        fail_threshold: Hallucination probability threshold.

    Returns:
        Same dict structure as screen_text().
    """
    source_docs = [gather_data_output] if gather_data_output else None
    return screen_text(
        text=memo_text,
        source_documents=source_docs,
        fail_threshold=fail_threshold,
    )
