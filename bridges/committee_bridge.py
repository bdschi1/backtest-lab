"""Bridge to multi-agent-investment-committee — consume T signal as alpha.

multi-agent-investment-committee produces a T signal per analysis:
    T = direction * entropy_adjusted_confidence
    T in [-1, +1]

This bridge lets backtest-lab consume those T signals as an
alpha source, either:
    1. Live: run the committee and feed the T signal into the engine
    2. Historical: load saved T signals from JSON and replay them
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl

from signals.base import Signal

logger = logging.getLogger(__name__)

_COMMITTEE_PATH = Path(
    "/Users/bdsm4/Library/CloudStorage/Dropbox/bds_repos/Tier_1/"
    "multi-agent-investment-committee"
)


def _ensure_import():
    path_str = str(_COMMITTEE_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class CommitteeTSignal(Signal):
    """Use multi-agent-IC's T signal as a backtest signal source.

    Loads a JSON file of historical T signals:
    [
        {"date": "2024-01-15", "ticker": "AAPL", "t_signal": 0.72},
        {"date": "2024-01-15", "ticker": "TSLA", "t_signal": -0.45},
        ...
    ]

    Args:
        signal_file: Path to JSON file with historical T signals.
    """

    def __init__(self, signal_file: str | Path):
        self._signal_file = Path(signal_file)
        self._signals: dict[date, dict[str, float]] = {}
        self._load()

    def _load(self):
        """Load T signals from JSON."""
        if not self._signal_file.exists():
            logger.warning("T signal file not found: %s", self._signal_file)
            return

        with open(self._signal_file) as f:
            data = json.load(f)

        for entry in data:
            d = date.fromisoformat(entry["date"])
            ticker = entry["ticker"]
            t = entry["t_signal"]
            if d not in self._signals:
                self._signals[d] = {}
            self._signals[d][ticker] = t

        logger.info(
            "Loaded %d T signals for %d dates",
            sum(len(v) for v in self._signals.values()),
            len(self._signals),
        )

    @property
    def name(self) -> str:
        return "Committee-T-Signal"

    @property
    def lookback_days(self) -> int:
        return 1  # no lookback needed — signals are pre-computed

    def generate_signals(
        self, prices: pl.DataFrame, current_date: date,
    ) -> dict[str, float]:
        """Return T signals for the current date."""
        return self._signals.get(current_date, {})


def load_doc_chunker():
    """Load the doc_chunker from multi-agent-IC for KB processing.

    This is the chunker you built — it processes PDFs (with page_range
    support) into token-bounded chunks for LLM consumption or
    risk document analysis.
    """
    _ensure_import()
    try:
        from tools.doc_chunker import process_uploads, format_kb_for_prompt
        return process_uploads, format_kb_for_prompt
    except ImportError:
        logger.warning("doc_chunker not available from multi-agent-IC")
        return None, None
