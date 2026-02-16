"""Tests for run log functionality — persistent JSON logging of backtest runs."""

from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from btconfig.run_config import DataConfig, OutputConfig, RunConfig
from btconfig.runner import run_backtest


def _run_in_tmp(tmp_dir: str) -> tuple:
    """Run a backtest writing results to a temp directory."""
    rc = RunConfig(
        initial_capital=100_000,
        data=DataConfig(
            universe=["AAPL", "MSFT", "GOOG"],
            start_date=date.today() - timedelta(days=365),
            end_date=date.today(),
            live=False,
        ),
        output=OutputConfig(
            markdown_tearsheet=False,
            results_dir=tmp_dir,
            run_log_path=f"{tmp_dir}/run_log.json",
        ),
    )
    return run_backtest(rc)


class TestRunLog:
    """Verify persistent run log."""

    def test_log_created_on_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_in_tmp(tmp)
            log_path = Path(tmp) / "run_log.json"
            assert log_path.exists()

    def test_log_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_in_tmp(tmp)
            log_path = Path(tmp) / "run_log.json"
            data = json.loads(log_path.read_text())
            assert isinstance(data, list)
            assert len(data) == 1

    def test_log_entry_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_in_tmp(tmp)
            log_path = Path(tmp) / "run_log.json"
            entries = json.loads(log_path.read_text())
            entry = entries[0]

            assert "run_id" in entry
            assert "timestamp" in entry
            assert "config" in entry
            assert "summary" in entry
            assert "output_dir" in entry
            assert "files" in entry

    def test_log_entry_config_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_in_tmp(tmp)
            entries = json.loads((Path(tmp) / "run_log.json").read_text())
            config = entries[0]["config"]

            assert config["initial_capital"] == 100_000
            assert "AAPL" in config["data"]["universe"]

    def test_log_entry_summary_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_in_tmp(tmp)
            entries = json.loads((Path(tmp) / "run_log.json").read_text())
            summary = entries[0]["summary"]

            assert "total_return_pct" in summary
            assert "sharpe_ratio" in summary

    def test_log_appended_on_subsequent_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run_in_tmp(tmp)
            _run_in_tmp(tmp)

            entries = json.loads((Path(tmp) / "run_log.json").read_text())
            assert len(entries) == 2
            # Different run_ids
            assert entries[0]["run_id"] != entries[1]["run_id"]

    def test_run_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_in_tmp(tmp)
            run_dir = summary["run_dir"]
            assert Path(run_dir).is_dir()

    def test_files_in_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_in_tmp(tmp)
            run_dir = Path(summary["run_dir"])

            # HTML tearsheet should always be there
            assert (run_dir / "tearsheet.html").exists()
            # Config YAML snapshot
            assert (run_dir / "config.yaml").exists()
