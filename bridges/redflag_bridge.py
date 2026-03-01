"""Bridge to redflag_ex1_analyst — pre-trade compliance gate.

redflag_ex1_analyst has 8 deterministic compliance rules:
    1. MNPI/tipping detection
    2. Steering indicators (expert call hours)
    3. Unverified claims
    4. Regulatory arbitrage (MiFID vs SEC)
    5. Defamation risk
    6. Momentum beta-neutral fallacy
    7. Optimization/crowding trap
    8. Endogenous risk

This bridge lets backtest-lab run a compliance check on
any research document before it influences trading signals.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bridges import resolve_tier1_repo

logger = logging.getLogger(__name__)

_REDFLAG_PATH = resolve_tier1_repo("redflag_ex1_analyst")


def _ensure_import():
    path_str = str(_REDFLAG_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def run_compliance_check(text: str) -> dict:
    """Run the redflag engine on a text document.

    Args:
        text: Research note, analyst memo, or LLM output to check.

    Returns:
        Dict with:
            - gate_decision: "PASS", "PM_REVIEW", or "AUTO_REJECT"
            - flags: list of detected red flags
            - score: total risk score
    """
    _ensure_import()
    try:
        from redflag_engine import RedFlagEngine
        engine = RedFlagEngine()
        result = engine.analyze(text)
        return result
    except ImportError:
        logger.warning("redflag_ex1_analyst not available")
        return {
            "gate_decision": "PASS",
            "flags": [],
            "score": 0,
            "error": "redflag engine not found at expected path",
        }


def check_document_file(file_path: str | Path) -> dict:
    """Load a document and run compliance check.

    Supports PDF, DOCX, TXT — uses redflag's document_loader.
    """
    _ensure_import()
    try:
        from document_loader import load_document
        from redflag_engine import RedFlagEngine

        text = load_document(str(file_path))
        engine = RedFlagEngine()
        return engine.analyze(text)
    except ImportError:
        logger.warning("redflag_ex1_analyst not available")
        return {"error": "redflag engine not found"}
