<!-- backtest-lab/README.md | Last updated: 2026-06-13 -->

# backtest-lab

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat&logo=polars&logoColor=white)
![tests](https://img.shields.io/badge/tests-381%20passing-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Event-driven backtesting engine for long/short equity strategies. Models spread, market impact, slippage, commission, and borrow cost at every fill. A risk manager gates every trade (position sizing, drawdown circuit breaker, exposure limits, ATR stops), and a regime detector adapts parameters in real time.

**Plain English:** It answers one question — what would this strategy have actually returned after real-world frictions? Signals only ever see point-in-time data; costs and risk checks apply before each trade, not as a post-hoc adjustment.

## Install

```
pip install -e .
pip install -e ".[ui]"          # CLI + TUI + Streamlit
pip install -e ".[bloomberg]"   # or .[ibkr] for optional live data
pip install -e ".[dev]"         # pytest, ruff, coverage
```

## Usage

```
python -m cli run --signal momentum --live --universe AAPL,MSFT,GOOG
python -m cli run --config examples/configs/ls_with_risk.yaml
python -m cli tui                       # interactive terminal UI
streamlit run ui/streamlit_app.py       # dashboard
```

All three interfaces build a `RunConfig` and call `run_backtest()`.

## What it does

- **Execution realism** — three fill tiers: naive mid → spread-aware → Almgren-Chriss market impact, with slippage / commission / borrow stacks
- **Enforced risk** — ATR trailing stops, drawdown circuit breaker (NORMAL / WARNING / HALTED), gross / net / single-name limits
- **Bias prevention** — structural look-ahead guard, walk-forward splits, deflated Sharpe, and a full Sharpe-inference suite (PSR, MinTRL, critical SR, FDR/FWER — Bailey & López de Prado)
- **Regime adaptation** — volatility or HMM regime detection mapped to parameter overrides
- **Bridges** — optional integration with ls-portfolio-lab, multi-agent-investment-committee, redflag, and fund-tracker-13f; each degrades gracefully if absent
- Output: timestamped HTML + Markdown tearsheets per run (`results/`, gitignored)

## Tests

```
pytest tests/ -v
```

## License

MIT
