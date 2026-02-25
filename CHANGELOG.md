# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-02-15

### Added
- Event-driven backtesting engine processing daily bars in strict sequential order
- Three signal types: momentum, mean reversion, and composite
- Three-tier execution modeling: naive (mid-price), realistic (spread + fixed slippage + per-share commission), and full (Almgren-Chriss market impact + volume slippage + tiered commission + tiered borrow)
- Risk manager gating every trade: ATR trailing stops, multi-constraint position sizer, drawdown circuit breaker (NORMAL/WARNING/HALTED), gross/net/single-name exposure limits
- Volatility targeting with configurable lookback, leverage bounds, and per-rebalance rescaling
- Regime detection via rolling realized vol thresholds and custom Gaussian HMM (no hmmlearn dependency)
- RegimeAdapter mapping regimes (LOW/NORMAL/ELEVATED/CRISIS) to parameter overrides
- Bias prevention: structural look-ahead guard, anchored + rolling walk-forward splits, deflated Sharpe ratio (Bailey & Lopez de Prado)
- Four position sizing modes: signal-weighted, fixed dollar, fixed shares, equal weight
- SPY benchmark comparison with beta, alpha, information ratio, up/down capture
- Portfolio-level risk decomposition: vol breakdown, exposure tracking, HHI concentration, beta-attributed vs alpha P&L
- Data provider abstraction (ABC) with Yahoo Finance (default), Bloomberg, and Interactive Brokers
- Static US equity/ETF ticker map (569 names) with fuzzy reverse lookup and validation
- Data source fallback chain with synthetic backfill for missing tickers
- Pydantic-validated RunConfig with nested sub-models and YAML config loading
- Self-contained HTML tearsheet, Markdown tearsheet with PNG charts, and dark-themed matplotlib chart generation
- Timestamped run directory output with append-only JSON run log
- CLI with full option set, interactive TUI (questionary), and Streamlit dashboard
- Integration bridges to ls-portfolio-lab, multi-agent-investment-committee, redflag_ex1_analyst, fund-tracker-13f
- 322 tests across 24 files
