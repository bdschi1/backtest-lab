# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## What This Is
Event-driven backtesting engine for long/short equity strategies. Models spread, market impact, slippage, commission, and borrow costs at every fill. Risk manager gates every trade through position sizing, drawdown circuit breakers, exposure limits, and ATR trailing stops. Regime detector adapts parameters in real time.

## Commands
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run (CLI)
python -m cli run --signal momentum --live --universe AAPL,MSFT,GOOG
python -m cli run --config examples/configs/ls_with_risk.yaml

# Run (Streamlit)
pip install -e ".[ui]"
streamlit run ui/streamlit_app.py

# Tests
pytest tests/ -v

# Lint
ruff check .
ruff format .
```

## Architecture
- `engine/` -- Event-driven loop, Portfolio, Position, DailySnapshot, TradeRecord
- `signals/` -- Signal ABC + momentum, mean reversion, composite implementations
- `execution/` -- Fill models (mid, spread, Almgren-Chriss impact), slippage, commission, borrow cost stacks
- `risk/` -- ATR stops, position sizer, drawdown circuit breaker, exposure limits, vol targeting, benchmark, risk decomposition. `risk_manager.py` composes all sub-checks
- `regime/` -- Volatility regime detector (threshold-based), HMM detector (custom EM), regime-to-parameter adapter
- `bias/` -- Lookahead guard, walk-forward splitter, overfit detector (deflated Sharpe), Sharpe inference (PSR, MinTRL, critical SR, power, FDR, FWER corrections — Bailey & Lopez de Prado)
- `data/` -- DataProvider ABC, Yahoo/Bloomberg/IB providers, provider factory, ticker map (569 names)
- `reports/` -- HTML tearsheet, matplotlib charts (dark theme), markdown tearsheet
- `btconfig/` -- Pydantic RunConfig, `runner.py` (single entry point for all interfaces), YAML loader
- `ui/` -- Streamlit dashboard, questionary-based TUI
- `bridges/` -- Integration bridges to ls-portfolio-lab, MAIC, redflag, fund-tracker-13f
- `cli.py` -- CLI entry point (argparse)

## Key Patterns
- All interfaces (CLI, TUI, Streamlit) build a `RunConfig` and call `run_backtest()` from `btconfig/runner.py`
- Engine processes bars sequentially: update prices -> risk state -> stops -> regime -> borrow -> signals -> risk gate -> vol-target -> execute -> snapshot
- Polars for all dataframe operations (no pandas). Pydantic for config validation
- Provider abstraction: Yahoo is default, Bloomberg/IB are optional installs via `pip install -e ".[bloomberg]"` or `".[ibkr]"`
- If a data provider fails, runner falls back to synthetic data automatically
- `results/` directory is gitignored; each run creates a timestamped subdirectory

## Testing Conventions
- 381 tests across 24 files in `tests/`
- Tests use synthetic/mock data exclusively -- no live API calls
- Run with `pytest tests/ -v`; config in `pyproject.toml` sets `testpaths` and `--tb=short`
- Bridges test graceful degradation when target repos are not installed
