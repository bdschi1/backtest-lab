# backtest-lab

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat&logo=polars&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-6001D2?style=flat&logo=yahoo&logoColor=white)
![Bloomberg](https://img.shields.io/badge/Bloomberg-000000?style=flat&logo=bloomberg&logoColor=white)
![Interactive Brokers](https://img.shields.io/badge/Interactive_Brokers-D71920?style=flat)

Event-driven backtesting engine for long/short equity strategies. Models spread, market impact, slippage, commission, and borrow costs at every fill. Risk manager gates every trade through position sizing, drawdown circuit breakers, exposure limits, and ATR trailing stops. Regime detector adapts parameters in real time.

This is a continually developed project. Features, interfaces, and test coverage expand over time as new research ideas and workflow needs arise.

**Key questions this project answers:**
- *What would this strategy have actually returned after real-world frictions?*
- *Does the signal survive transaction costs, slippage, and risk constraints?*

---

## Purpose

backtest-lab exists to answer a specific question: **what would this strategy have actually returned after real-world frictions?**

Many backtesting frameworks report gross returns and ignore the execution costs, risk constraints, and structural biases that separate a research idea from a deployable strategy. backtest-lab treats these as first-class concerns:

- **Execution realism** -- three tiers of fill modeling from naive mid-price to Almgren-Chriss market impact, with spread, slippage, commission, and borrow cost stacks
- **Enforced risk management** -- not just metrics after the fact, but real-time constraints that reject or size down trades before execution (ATR stops, drawdown circuit breaker, exposure limits)
- **Bias prevention** -- structural look-ahead guards that physically prevent signals from accessing future data, walk-forward validation, and overfit detection via deflated Sharpe ratio
- **Regime adaptation** -- volatility regime detection (threshold-based or HMM) that adjusts risk parameters and gross exposure dynamically

The target user is someone running a long/short equity book who wants to stress-test signal ideas with realistic frictions before allocating capital.

---

## Background

The engine processes each trading day bar-by-bar in strict sequential order. Signals only see data up to the current date. Risk checks run before every trade. Costs accrue at execution time, not as a post-hoc adjustment.

Key design choices:

- **Polars** for price data (columnar, fast filtering by date)
- **Pydantic** for configuration validation (all inputs are typed and bounded)
- **No pandas dependency** -- Polars handles all dataframe operations
- **No external ML deps for regime detection** -- custom Gaussian HMM with EM fitting (no hmmlearn)
- **Provider abstraction** -- Yahoo Finance ships free; Bloomberg and Interactive Brokers plug in via the same ABC

---

## Setup

```bash
# Core install
pip install -e .

# With all UI interfaces
pip install -e ".[ui]"

# Optional data providers
pip install -e ".[bloomberg]"   # Bloomberg Terminal via blpapi
pip install -e ".[ibkr]"        # Interactive Brokers via ib_insync

# Development
pip install -e ".[dev]"         # pytest, ruff, coverage
```

Requires Python 3.11+.

---

## How to Use

backtest-lab provides three interfaces that all share the same configuration model and execution path. Every interface builds a `RunConfig` and calls `run_backtest()`.

### CLI

```bash
# Basic momentum backtest with synthetic data
python -m cli run --signal momentum

# Live data with fixed-dollar sizing and vol targeting
python -m cli run --signal momentum --live \
  --universe AAPL,MSFT,GOOG,AMZN,TSLA \
  --sizing-mode fixed_dollar --fixed-dollar 50000 \
  --vol-target 10

# Full pipeline: risk + regime + custom rebalance
python -m cli run --signal composite --live --risk --regime \
  --rebalance weekly --capital 5000000

# Run from a saved YAML config
python -m cli run --config examples/configs/ls_with_risk.yaml

# Save your CLI config for later reuse
python -m cli run --signal momentum --live --risk --save-config my_run.yaml

# List available data providers
python -m cli providers

# Run the 3-tier execution comparison example
python -m cli example
```

### Interactive TUI

```bash
python -m cli tui
```

Walks through 10 steps: signal selection, universe, execution tier, capital, position sizing mode, volatility target, risk controls, regime detection, output options, and a summary confirmation before running.

### Streamlit Dashboard

```bash
streamlit run ui/streamlit_app.py
```

Full sidebar controls for every parameter. Results display inline with summary metrics, embedded HTML tearsheet, and download buttons for HTML, Markdown, and YAML config exports.

### YAML Configuration

Any run can be driven from a YAML file. Example configs are in `examples/configs/`:

```bash
python -m cli run --config examples/configs/ls_with_risk.yaml
```

---

## CLI Reference

Run `python -m cli run --help` for the full flag list. Key options:

```bash
python -m cli run --signal momentum --live --universe AAPL,MSFT,GOOG --risk --regime
python -m cli run --config examples/configs/ls_with_risk.yaml
```

Supports four sizing modes (`signal`, `fixed_dollar`, `fixed_shares`, `equal_weight`), optional volatility targeting, three execution tiers (`mid`, `spread`, `impact`), and YAML config files for reproducibility.

---

## Output

Each run creates a timestamped directory under `results/` (gitignored) with an HTML tearsheet, Markdown tearsheet, chart PNGs, and a frozen YAML config. A persistent `run_log.json` accumulates summary statistics across runs.

---

## Data Providers

yfinance is the default (free, daily OHLCV). Bloomberg and Interactive Brokers are optional installs (`pip install -e ".[bloomberg]"` or `".[ibkr]"`) with real bid/ask data. If a provider fails, the runner falls back to synthetic data automatically.

---

## Architecture

```
backtest-lab/
  engine/
    backtest.py             Event-driven loop, Portfolio, Position, DailySnapshot, TradeRecord
  signals/
    base.py                 Signal ABC
    momentum.py             MomentumSignal, MeanReversionSignal, CompositeSignal
  execution/
    fill_model.py           MidPriceFill, SpreadAwareFill, MarketImpactFill (Almgren-Chriss)
    slippage.py             ZeroSlippage, FixedSlippage, VolumeSlippage
    commission.py           ZeroCommission, PerShareCommission, TieredCommission
    borrow.py               ZeroBorrow, FixedBorrow, TieredBorrow (GC/Warm/Special/HTB)
  risk/
    stop_loss.py            ATRTrailingStop with hard loss cap
    position_sizer.py       Multi-constraint: equity %, ADV %, notional cap
    drawdown_control.py     Circuit breaker: NORMAL -> WARNING -> HALTED
    exposure_limits.py      Gross/net/single-name/sector caps
    risk_manager.py         Composes all risk sub-checks, gates every trade
    vol_target.py           VolTargetScaler for portfolio-level vol targeting
    benchmark.py            SPY benchmark comparison (beta, alpha, info ratio, capture)
    risk_decomposition.py   Position/theme/portfolio risk decomposition
  regime/
    detector.py             VolatilityRegimeDetector (LOW/NORMAL/ELEVATED/CRISIS)
    hmm_detector.py         Gaussian HMM with custom EM (no hmmlearn dependency)
    adapter.py              RegimeAdapter: regime -> parameter overrides
  bias/
    lookahead_guard.py      Structural prevention of look-ahead bias
    walk_forward.py         Anchored + rolling walk-forward splits
    overfit_detector.py     Sharpe decay + deflated Sharpe ratio (Bailey & Lopez de Prado)
    sharpe_inference.py     Full Sharpe inference: PSR, MinTRL, critical SR, power, FDR, FWER
    sharpe_statistics.py    Sharpe CI with autocorrelation adjustment, PSR, MinTRL
    rademacher.py           Rademacher complexity bounds for data snooping
  data/
    provider.py             DataProvider ABC
    yahoo_provider.py       Free EOD data via yfinance
    bloomberg_provider.py   Bloomberg Terminal (optional, requires blpapi)
    ib_provider.py          Interactive Brokers (optional, requires ib_insync)
    provider_factory.py     Auto-detect installed providers
    ticker_map.py           US equity/ETF ticker map (569 names: S&P 500 + non-index + ETFs)
  reports/
    tearsheet.py            Self-contained HTML with inline SVG charts
    charts.py               Matplotlib PNG generation (dark theme)
    markdown_tearsheet.py   Markdown tearsheet with chart references
  btconfig/
    run_config.py           Pydantic RunConfig with nested sub-models
    runner.py               run_backtest(): single entry point for all interfaces
    yaml_loader.py          YAML config loading
  ui/
    streamlit_app.py        Streamlit dashboard
    tui.py                  Interactive terminal UI (questionary-based)
  bridges/
    portfolio_lab_bridge.py   Factor models from ls-portfolio-lab
    committee_bridge.py       T signals from multi-agent-investment-committee
    redflag_bridge.py         Compliance gate from redflag_ex1_analyst
    fund_tracker_bridge.py    13F conviction signals from fund-tracker-13f
    kb_risk_bridge.py         PDF chunking for risk knowledge base
  cli.py                    CLI entry point (argparse)
  examples/
    momentum_backtest.py    3-tier execution comparison
    configs/                YAML example configs
  tests/                    381 tests across 24 files
```

---

## Engine Flow

Each bar executes in this order:

1. **Update prices** -- mark all positions to market
2. **Risk state** -- update drawdown HWM, ATR, circuit breaker
3. **Stop-loss check** -- force-close any positions that breached stops
4. **Regime detection** -- classify market vol, adapt parameters
5. **Accrue borrow costs** -- daily cost on short positions
6. **Generate signals** -- point-in-time only, no future data
7. **Risk gate** -- approve/reject/size-down each trade
8. **Vol-target scaling** -- scale all targets to match portfolio vol target
9. **Execute** -- fill model + slippage + commission
10. **Record snapshot** -- equity, costs, regime, risk state

---

## Execution Realism

Three tiers of execution modeling:

| Tier | Fill | Slippage | Commission | Borrow | Use Case |
|------|------|----------|------------|--------|----------|
| Naive | MidPrice (close) | Zero | Zero | Zero | Quick prototyping |
| Realistic | SpreadAware | Fixed (5 bps) | PerShare ($0.005) | Fixed | Strategy evaluation |
| Full | MarketImpact | Volume-based | Tiered (IB) | Tiered (HTB) | Production simulation |

Market impact uses the Almgren-Chriss square-root model: `impact = eta x sigma x sqrt(shares/ADV)`.

---

## Risk Management

The risk manager runs before every trade and can reject or size down:

- **ATR Trailing Stop** -- trails high-water mark by N x ATR, with hard loss cap
- **Position Sizer** -- constrains to min(signal x equity%, ADV%, notional cap)
- **Drawdown Controller** -- circuit breaker with 3 states:
  - NORMAL: full trading
  - WARNING (-5%): positions scaled to 50%
  - HALTED (-15%): no new trades until recovery
- **Exposure Limits** -- enforces gross/net/single-name caps, scales trades to fit

---

## Analytics

Every backtest computes trade-level metrics (win rate, profit factor, payoff ratio, drawdown duration) and portfolio-level analytics (Sharpe, Sortino, max drawdown, Calmar). When live data is available, SPY benchmark comparison runs automatically (beta, alpha, information ratio, up/down capture).

Post-backtest risk decomposition covers portfolio-level vol (realized, downside, systematic vs. idiosyncratic), exposure tracking (gross/net/single-name), and concentration metrics (HHI, return decomposition into beta-attributed vs. alpha P&L). All metrics appear in the HTML and Markdown tearsheets.

---

## Regime Detection

Two detectors available:

- **VolatilityRegimeDetector** -- rolling realized vol vs fixed thresholds (fast, interpretable)
- **HMMRegimeDetector** -- Gaussian HMM with custom EM fitting (more sophisticated, no external deps)

Regime classifications: LOW (<12% ann vol), NORMAL (12-20%), ELEVATED (20-30%), CRISIS (>30%).

The **RegimeAdapter** maps regimes to parameter overrides:
- LOW: +20% gross, wider stops
- NORMAL: standard parameters
- ELEVATED: -30% gross, tighter stops, higher signal threshold
- CRISIS: -60% gross, minimal new positions

---

## Bias Prevention

- **LookaheadGuard** -- structurally prevents signals from seeing future data (not by convention, by code)
- **WalkForwardSplitter** -- anchored or rolling train/test splits (default: 504/63 days)
- **OverfitDetector** -- flags if in-sample Sharpe decays >50% out-of-sample; computes deflated Sharpe ratio adjusted for multiple testing (Bailey & Lopez de Prado)
- **Sharpe Inference** (`bias/sharpe_inference.py`) -- rigorous statistical testing for Sharpe ratios:
  - **Probabilistic Sharpe Ratio (PSR):** P(true SR > benchmark) accounting for skewness, kurtosis, and autocorrelation (Bailey & Lopez de Prado, 2014)
  - **Minimum Track Record Length (MinTRL):** minimum observations needed for statistical significance
  - **Critical Sharpe Ratio:** threshold SR for rejecting H₀ at level α, adjusted for multiple testing
  - **Statistical Power:** P(reject H₀ | true SR = SR₁) for sample size planning
  - **False Discovery Rate:** posterior and observed FDR for strategy testing pipelines
  - **FWER Corrections:** Bonferroni, Šidák, and Holm step-down adjustments for multiple comparisons
  - **Expected Maximum SR:** exact E[max] of K normals via Gauss-Hermite quadrature (replaces √(2·ln(K)) approximation)

In plain terms, these tools answer: is this backtest result statistically real, or could it be luck? They adjust for how many strategies were tested, how long the track record is, and whether the return distribution is well-behaved.

---

## Bridges

Integration bridges to other repos in the same ecosystem. All gracefully degrade if the target repo is not installed:

| Bridge | Source Repo | What It Does |
|--------|-------------|--------------|
| portfolio_lab_bridge | ls-portfolio-lab | Factor exposure (CAPM/FF3/FF4) on backtest returns |
| committee_bridge | multi-agent-investment-committee | Use pre-computed T signals as alpha source |
| redflag_bridge | redflag_ex1_analyst | Pre-trade compliance gate (MNPI, tipping, defamation) |
| fund_tracker_bridge | fund-tracker-13f | 13F conviction signals as universe filter |
| kb_risk_bridge | multi-agent-investment-committee | PDF chunking for risk document knowledge base |

---

## Data Providers

| Provider | Bid/Ask | Setup |
|----------|---------|-------|
| Yahoo Finance | No (estimated from range) | Default, free, always available |
| Bloomberg | Yes | `pip install -e ".[bloomberg]"`, requires Terminal |
| Interactive Brokers | Yes | `pip install -e ".[ibkr]"`, requires TWS/Gateway |

All providers share the same `DataProvider` ABC. `data/ticker_map.py` provides a static 569-ticker US equity universe (S&P 500 + notable non-index + major ETFs) with name/sector lookup functions.

---

## Tests

```bash
python -m pytest tests/ -v
```

381 tests across 24 files covering engine, signals, execution models, risk management, regime detection, bias prevention, configuration, run logging, tearsheet generation, position sizing, volatility targeting, benchmark comparison, risk decomposition, ticker mapping, and data source tracking.

---

## Contributing

Under active development. Contributions welcome — areas for improvement include additional signal types, execution cost models, risk overlays, and reporting formats.

---

***Curiosity compounds. Rigor endures.***

## License

MIT

