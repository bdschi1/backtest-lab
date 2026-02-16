"""Tests for reports/markdown_tearsheet.py and reports/charts.py."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from btconfig.run_config import DataConfig, OutputConfig, RunConfig
from btconfig.runner import run_backtest


def _run_with_markdown(tmp_dir: str) -> tuple:
    """Run a backtest with markdown output into a temp directory."""
    rc = RunConfig(
        initial_capital=100_000,
        data=DataConfig(
            universe=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
            start_date=date.today() - timedelta(days=365),
            end_date=date.today(),
            live=False,
        ),
        output=OutputConfig(
            markdown_tearsheet=True,
            results_dir=tmp_dir,
            run_log_path=f"{tmp_dir}/run_log.json",
        ),
    )
    return run_backtest(rc)


class TestMarkdownTearsheet:
    """Test markdown tearsheet generation."""

    def test_markdown_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            md_path = Path(run_dir) / "tearsheet.md"
            assert md_path.exists()

    def test_markdown_has_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            md_path = Path(run_dir) / "tearsheet.md"
            content = md_path.read_text()
            assert len(content) > 100
            assert "# Backtest:" in content

    def test_markdown_has_summary_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            content = (Path(run_dir) / "tearsheet.md").read_text()
            assert "## Summary" in content
            assert "Total Return" in content
            assert "Sharpe" in content

    def test_markdown_has_chart_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            content = (Path(run_dir) / "tearsheet.md").read_text()
            assert "![Equity Curve]" in content
            assert "![Drawdown]" in content
            assert "![Rolling Sharpe]" in content

    def test_png_charts_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = Path(summary["run_dir"])
            assert (run_dir / "equity_curve.png").exists()
            assert (run_dir / "drawdown.png").exists()
            assert (run_dir / "rolling_sharpe.png").exists()
            assert (run_dir / "monthly_returns.png").exists()

    def test_png_files_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = Path(summary["run_dir"])
            for name in ["equity_curve.png", "drawdown.png", "rolling_sharpe.png"]:
                png = run_dir / name
                assert png.stat().st_size > 1000, f"{name} is too small"

    def test_monthly_returns_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            content = (Path(run_dir) / "tearsheet.md").read_text()
            assert "## Monthly Returns" in content
            assert "Jan" in content
            assert "Annual" in content

    def test_cost_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            content = (Path(run_dir) / "tearsheet.md").read_text()
            assert "## Execution Costs" in content

    def test_trade_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = summary["run_dir"]
            content = (Path(run_dir) / "tearsheet.md").read_text()
            assert "## Trade Log" in content


class TestHtmlTearsheetStillWorks:
    """Verify HTML tearsheet is also generated."""

    def test_html_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            html_path = summary.get("tearsheet_html")
            assert html_path is not None
            assert Path(html_path).exists()

    def test_config_yaml_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, summary = _run_with_markdown(tmp)
            run_dir = Path(summary["run_dir"])
            assert (run_dir / "config.yaml").exists()
