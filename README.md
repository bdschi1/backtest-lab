# backtest-lab

Event-driven backtesting engine for long/short equity strategies. Models spread, market impact, slippage, commission, and borrow costs at every fill. Risk manager gates every trade through position sizing, drawdown circuit breakers, exposure limits, and ATR trailing stops. Regime detector adapts parameters in real time.

This is a continually developed project. Features, interfaces, and test coverage expand over time as new research ideas and workflow needs arise.

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

```
python -m cli run [OPTIONS]

Signal:
  --signal          momentum | mean_reversion | composite (default: momentum)

Data:
  --universe        Comma-separated tickers (default: 10 large caps)
  --start           Start date YYYY-MM-DD (default: 3 years ago)
  --end             End date YYYY-MM-DD (default: today)
  --live            Use live yfinance data (default: synthetic)
  --provider        yahoo | bloomberg | ib (default: yahoo)

Portfolio:
  --capital         Initial capital (default: 1,000,000)
  --rebalance       daily | weekly | monthly (default: weekly)
  --max-position-pct  Max position size as % of equity (default: 10)

Position Sizing:
  --sizing-mode     signal | fixed_dollar | fixed_shares | equal_weight
  --fixed-dollar    Dollar amount per position (with fixed_dollar mode)
  --fixed-shares    Share count per position (with fixed_shares mode)

Volatility Targeting:
  --vol-target      Target annual vol %; enables vol targeting when set
  --vol-lookback    Lookback days for realized vol (default: 63)

Execution:
  --fill            mid | spread | impact (default: mid)

Risk & Regime:
  --risk            Enable risk manager
  --regime          Enable regime detection

Output:
  --tearsheet       Path for HTML tearsheet
  --results-dir     Base results directory (default: results)
  --config          Load from YAML config file
  --save-config     Save current config to YAML file
```

---

## Position Sizing

Four sizing modes control how target positions are computed:

| Mode | How It Works |
|------|-------------|
| `signal` | Signal score x max position % x equity (default) |
| `fixed_dollar` | User specifies dollar amount per position; shares computed from price |
| `fixed_shares` | User specifies share count per position; direction from signal |
| `equal_weight` | Equity divided equally across all active positions |

---

## Volatility Targeting

When enabled, portfolio positions are scaled so realized volatility tracks the target. Uses trailing realized vol over a configurable lookback window and rescales gross exposure at each rebalance.

- `target_annual_vol_pct`: desired annualized vol (1-50%, default 10%)
- `lookback_days`: trailing window for realized vol (10-252, default 63)
- `max_leverage`: cap on upward scaling (1-10x, default 3x)
- `min_leverage`: floor on downward scaling (0.01-1x, default 0.1x)

Scale factor: `target_vol / realized_vol`, clamped to [min_leverage, max_leverage].

---

## Output and Run Logging

Each backtest creates a timestamped directory under `results/`:

```
results/
  run_log.json                              # Persistent JSON log of all runs
  2026-02-15_153022_123456_momentum/
    tearsheet.html                           # Self-contained HTML report
    tearsheet.md                             # Markdown report with PNG references
    equity_curve.png                         # Dark-themed matplotlib charts
    drawdown.png
    rolling_sharpe.png
    monthly_returns.png
    config.yaml                              # Frozen config for reproducibility
```

The `run_log.json` file is an append-only JSON array. Each entry captures the full config snapshot, summary statistics, and file paths. It is never committed to git -- `results/` is gitignored.

The Markdown tearsheet contains the same information as the HTML version: summary statistics table, chart image references, monthly returns table, cost breakdown, and trade log.

---

## Data Source and Fallback

yfinance is the default data provider. It provides daily OHLCV data for any ticker listed on NYSE, NASDAQ, or major global exchanges.

**Limitations:** No intraday data. No real-time quotes. No bid/ask spreads (estimated from daily range). Data may be delayed 15-20 minutes. Adjusted close accounts for splits and dividends. Some tickers may be unavailable.

Bloomberg and Interactive Brokers providers are available as optional installs and provide real bid/ask data.

**Fallback chain:** If a data provider fails entirely, the runner automatically falls back to synthetic data so the backtest still completes. If a provider returns data for some tickers but not others, the missing tickers are backfilled with synthetic data and a warning is logged. This ensures runs never fail silently due to transient API issues or unsupported tickers.

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
  tests/                    322 tests across 24 files
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

## Performance Analytics

Every backtest computes a full suite of trade-level and portfolio-level metrics:

**Trade Analytics:**

| Metric | Description |
|--------|-------------|
| Win Rate | Percentage of profitable round-trip trades |
| Slugging Pct | Average winning trade / average losing trade |
| Profit Factor | Gross profit / gross loss |
| Payoff Ratio | Average win / average loss |
| Best / Worst Trade | Largest single-trade P&L |
| Avg Trade | Mean realized P&L per round trip |

**Drawdown Duration:**

| Metric | Description |
|--------|-------------|
| Max DD Duration | Longest peak-to-recovery period (trading days) |
| Max DD Recovery | Longest trough-to-recovery period (trading days) |
| Avg DD Duration | Mean drawdown episode length |
| Current DD Duration | Active drawdown length (0 if at HWM) |

---

## Benchmark Comparison

When live data is available, the runner automatically fetches SPY for the same date range and computes benchmark-relative metrics:

| Metric | Description |
|--------|-------------|
| Relative Return | Strategy cumulative return minus SPY |
| Beta | CAPM regression slope (covariance / variance) |
| Alpha | Jensen's alpha, annualized |
| Information Ratio | Annualized active return / tracking error |
| Tracking Error | Annualized standard deviation of active returns |
| Up / Down Capture | Strategy participation in benchmark up and down periods |
| Correlation | Pearson correlation to benchmark daily returns |

If SPY data is unavailable, the backtest still completes — benchmark metrics are omitted.

---

## Risk Decomposition

Post-backtest risk decomposition operates at three levels:

**Portfolio-level volatility:**
- Annualized vol, downside vol (semi-deviation)
- Rolling 21-day and 63-day realized vol with current, average, max, and min
- Vol decomposition into systematic (beta) and idiosyncratic components

**Exposure tracking:**
- Gross, net, long, and short exposure as % of equity (time-series averages and finals)
- Position count statistics (average, max)
- Beta-net exposure (sum of position_beta × position_weight)

**Concentration and decomposition:**
- HHI (Herfindahl-Hirschman Index) for position concentration
- Top-1 and top-5 name weights
- Beta-attributed P&L vs. alpha P&L (return decomposition)
- Idiosyncratic return ratio (% of variance unexplained by market)
- Idiosyncratic volatility and idiosyncratic Sharpe ratio

In plain terms: how concentrated is the portfolio, how much of the return came from market exposure vs. stock-picking, and is the stock-specific return consistent enough to be meaningful.

All risk decomposition metrics appear in the HTML tearsheet, Markdown tearsheet, and Streamlit dashboard.

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

### Bloomberg Setup

Install the optional dependency and ensure a Bloomberg Terminal or B-PIPE session is running:

```bash
pip install -e ".[bloomberg]"
```

The provider connects lazily on first data request (default: `localhost:8194`). Tickers are automatically converted to Bloomberg format (`AAPL` becomes `AAPL US Equity`). Fields: PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, BID, ASK. To use from the CLI:

```bash
python -m cli run --live --provider bloomberg --signal momentum
```

### Interactive Brokers Setup

Install the optional dependency and ensure TWS or IB Gateway is running:

```bash
pip install -e ".[ibkr]"
```

Default connection: `127.0.0.1:7497` (TWS paper trading). For live trading use port `7496`; for IB Gateway use `4001`/`4002`. The provider supports daily OHLCV via `reqHistoricalData` and real-time snapshots via `reqMktData`. Bid/ask data is fetched as a separate BID_ASK bar request. To use from the CLI:

```bash
python -m cli run --live --provider ib --signal momentum
```

Both providers share the same `DataProvider` ABC and can be swapped with a single flag change. If a provider is unavailable at runtime, the factory falls back to Yahoo Finance automatically.

### Ticker Mapping

`data/ticker_map.py` provides a static US equity universe (569 tickers) covering:

- **All ~503 S&P 500 constituents** (including dual-class shares)
- **~40 notable non-index stocks** (PLTR, UBER, RIVN, GME, BABA, TSM, NVO, ARM, etc.)
- **~25 major ETFs** (SPY, QQQ, sector SPDRs, GLD, TLT, HYG, IBIT, etc.)

Each entry maps ticker → company name + GICS sector (or "ETF"). Functions: `ticker_to_name()`, `ticker_to_sector()`, `name_to_ticker()` (fuzzy reverse lookup), `get_sector_map()`, `validate_tickers()`.

This is a hardcoded, point-in-time snapshot — no API call on every run. Index rebalances require manual update.

**Known issue:** The ticker LLY (Eli Lilly & Co.) may resolve to a different security in some data providers outside yfinance. yfinance correctly maps LLY to Eli Lilly. When using Bloomberg or IB, verify the resolved instrument matches Eli Lilly (NYSE: LLY, ISIN: US5324571083). This is documented in the `_KNOWN_TICKER_ISSUES` registry and `validate_tickers()` warns when LLY is in the universe.

### Data Source Transparency

Every backtest run records its data provenance. The runner tracks:
- Provider used (Yahoo Finance, Bloomberg, IB, or Synthetic fallback)
- Data frequency (daily, intraday, tick)
- Date range fetched
- Tickers resolved vs. failed
- Provider-specific limitations

This metadata appears in three places:
1. **Streamlit banner** — colored info box at top of results showing provider, date range, ticker count
2. **Tearsheets** (HTML + Markdown) — data source line at top of report
3. **Run log** (`run_log.json`) — `data_source` object in each entry

The provider registry supports Yahoo Finance (always available), Bloomberg Terminal, Interactive Brokers, and planned stubs for Refinitiv/LSEG, Polygon.io, and Databento.

---

## Training Data Generation

The structured output of each backtest run (config YAML, summary JSON, trade log, cost attribution) can serve as labeled training data for financial language models. Each entry in `run_log.json` maps a complete input configuration to observed outcomes with full cost decomposition — the kind of structured evaluation that requires domain expertise to produce manually.

A parameter sweep across signal types, sizing modes, risk settings, and regime configurations generates supervised examples at scale. Each example captures:

- **Input**: full `RunConfig` snapshot (signal parameters, sizing mode, risk thresholds, execution tier, universe, date range)
- **Output**: summary statistics (Sharpe, Sortino, max drawdown, Calmar, total costs, stops triggered, trades rejected), cost-attributed returns, and trade-level detail
- **Context**: why a strategy underperformed (e.g., risk manager halted trading at -15% drawdown, transaction costs exceeded gross alpha, vol regime shifted from NORMAL to CRISIS mid-backtest)

The run log accumulates over time as a growing corpus. This is distinct from generic financial text data — every entry is a structured input-to-outcome pair with ground truth from a simulation that enforces real-world constraints.

---

## Tests

```bash
python -m pytest tests/ -v
```

322 tests across 24 files covering engine, signals, execution models, risk management, regime detection, bias prevention, configuration, run logging, tearsheet generation, position sizing, volatility targeting, benchmark comparison, risk decomposition, ticker mapping, and data source tracking.

---

## Status

This project is under active, ongoing development. The core engine, risk management, and execution modeling are stable. New signal types, additional risk overlays, and extended reporting are added as research needs evolve.

---

## License

MIT

---

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)

![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat&logo=polars&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-6001D2?style=flat&logo=yahoo&logoColor=white)
![Bloomberg](https://img.shields.io/badge/Bloomberg-000000?style=flat&logo=bloomberg&logoColor=white)
![Interactive Brokers](https://img.shields.io/badge/Interactive_Brokers-D71920?style=flat)
