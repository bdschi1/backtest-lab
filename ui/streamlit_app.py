"""Streamlit dashboard for backtest-lab.

Run with:
    streamlit run ui/streamlit_app.py

All configuration is controlled via the sidebar. Results appear in the
main area after clicking RUN.  The tearsheet HTML is generated to a temp
file, read back, and embedded inline via st.components.v1.html().
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from btconfig.run_config import (
    RunConfig,
    SignalConfig,
    DataConfig,
    FillConfig,
    ExecutionConfig,
    SizingConfig,
    VolTargetConfig,
    RiskConfig,
    RegimeConfig,
    OutputConfig,
)
from btconfig.runner import run_backtest

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="backtest-lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal dark-friendly CSS overrides
st.markdown(
    """
    <style>
    /* tighten sidebar spacing */
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    /* metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 31, 38, 0.6);
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 6px;
        padding: 12px 16px 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "engine" not in st.session_state:
    st.session_state.engine = None
if "tearsheet_html" not in st.session_state:
    st.session_state.tearsheet_html = None
if "run_config" not in st.session_state:
    st.session_state.run_config = None

# ===== SIDEBAR =============================================================
with st.sidebar:
    st.title("backtest-lab")
    st.caption("Configure and run a backtest")

    with st.expander("About backtest-lab"):
        st.markdown("""
**What it is**

An institutional-grade backtesting engine for equity long/short strategies.
It simulates daily portfolio rebalancing with realistic execution costs,
multi-layer risk management, and benchmark-relative performance attribution.

**Why it exists**

To bridge the gap between quantitative signal research and portfolio-level
risk analysis. A PM or risk manager can evaluate a signal's P&L, cost drag,
drawdown behavior, and idiosyncratic return profile without writing code.
An AI engineer can use the structured output as labeled training data for
financial language models.

**What it does**

Takes a ticker universe + signal definition and runs a daily event-loop
simulation:

- **Signals**: Momentum, mean-reversion, or composite (weighted blend)
- **Execution**: Three-tier fill models (mid, spread-aware, market impact)
  with five cost components (commission, spread, slippage, market impact,
  borrow)
- **Risk management**: ATR trailing stops, position sizing constraints,
  drawdown circuit breaker (NORMAL → WARNING → HALTED), gross/net/single-name
  exposure limits
- **Vol targeting**: Scale portfolio to a target annualized volatility
- **Regime detection**: Classify market conditions (LOW/NORMAL/ELEVATED/CRISIS)
  and adapt parameters automatically
- **Analytics**: Trade-level P&L (win rate, slugging, profit factor),
  drawdown duration, SPY benchmark comparison (beta, alpha, information ratio,
  up/down capture), full risk decomposition (portfolio vol, exposure tracking,
  concentration, beta-attributed vs. alpha P&L, idiosyncratic Sharpe)

**Data**

Yahoo Finance by default (free, daily adjusted close for any NYSE/NASDAQ
ticker). Plug-in adapters for Bloomberg Terminal and Interactive Brokers
provide intraday, real-time, and bid/ask data when available. Planned support
for Refinitiv, Polygon.io, and Databento.

**How to run**

Three interfaces, same engine:
- **CLI**: `python -m cli run --universe AAPL,MSFT --live --signal momentum`
- **TUI**: `python -m cli tui` (guided step-by-step)
- **Streamlit**: `streamlit run ui/streamlit_app.py` (this dashboard)

All share the same `RunConfig → run_backtest()` pipeline. Results are saved
to a timestamped directory under `results/` with HTML tearsheet, Markdown
tearsheet, config snapshot, and run log entry.

**Architecture** (for engineers)

Pydantic config → BacktestEngine (event-loop with Position/Trade dataclasses)
→ RiskManager (composes stop-loss, sizer, drawdown, exposure sub-checks) →
CostModel (5-component) → Tearsheet (HTML/MD/Streamlit). Extensible signal
interface, provider abstraction (DataProvider ABC), HMM regime detection,
walk-forward validation, and overfit detection (deflated Sharpe ratio).
""")

    # ---- 1. Signal ----------------------------------------------------------
    st.header("Signal")

    signal_type = st.selectbox(
        "Signal type",
        options=["momentum", "mean_reversion", "composite"],
        index=0,
    )

    # Momentum params (shown for momentum and composite)
    if signal_type in ("momentum", "composite"):
        st.subheader("Momentum parameters")
        mom_formation = st.slider(
            "Formation days", min_value=20, max_value=504, value=126, step=1,
        )
        mom_skip = st.slider(
            "Skip days", min_value=0, max_value=63, value=21, step=1,
        )
        mom_zscore = st.slider(
            "Z-score clip", min_value=1.0, max_value=5.0, value=3.0, step=0.1,
        )
    else:
        mom_formation, mom_skip, mom_zscore = 126, 21, 3.0

    # Mean-reversion params (shown for mean_reversion and composite)
    if signal_type in ("mean_reversion", "composite"):
        st.subheader("Mean-reversion parameters")
        mr_lookback = st.slider(
            "Lookback days", min_value=5, max_value=252, value=21, step=1,
        )
        mr_zscore = st.slider(
            "MR Z-score clip", min_value=1.0, max_value=5.0, value=3.0, step=0.1,
        )
    else:
        mr_lookback, mr_zscore = 21, 3.0

    # Composite weights
    if signal_type == "composite":
        st.subheader("Composite weights")
        comp_mom_w = st.slider(
            "Momentum weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        )
        comp_mr_w = round(1.0 - comp_mom_w, 2)
        st.caption(f"Mean-reversion weight: {comp_mr_w}")
    else:
        comp_mom_w, comp_mr_w = 0.6, 0.4

    st.divider()

    # ---- 2. Universe --------------------------------------------------------
    st.header("Universe")

    default_tickers = "AAPL\nMSFT\nGOOG\nAMZN\nTSLA\nNVDA\nMETA\nJPM\nV\nJNJ"
    tickers_raw = st.text_area(
        "Tickers (one per line or comma-separated)",
        value=default_tickers,
        height=150,
    )

    with st.expander("Connect a Professional Data Source"):
        st.markdown(
            """
**Yahoo Finance** (active) — free, daily adjusted close

---

**Bloomberg Terminal** — `pip install blpapi`
> Intraday pricing, real-time, bid/ask spreads, corporate actions, reference data

**Interactive Brokers** — `pip install ib_insync`
> Real-time streaming, historical intraday, bid/ask, options chain

**Refinitiv / LSEG** — *(coming soon)*
> Tick data, reference data, economic indicators

**Polygon.io** — *(coming soon)*
> REST API, WebSocket streaming, trades & quotes, 15+ years history

**Databento** — *(coming soon)*
> Tick-by-tick, normalized across exchanges, futures/options

---

| Capability | Yahoo | Bloomberg | IB |
|---|---|---|---|
| Daily OHLCV | Yes | Yes | Yes |
| Intraday | — | Yes | Yes |
| Real-time | — | Yes | Yes |
| Bid/Ask | — | Yes | Yes |
| Cost | Free | $$$$ | $$ |
"""
        )

    st.divider()

    # ---- 3. Date Range ------------------------------------------------------
    st.header("Date Range")

    default_end = date.today()
    default_start = default_end - timedelta(days=3 * 365)

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start", value=default_start)
    with col_end:
        end_date = st.date_input("End", value=default_end)

    use_live = st.checkbox("Use live data", value=False)
    provider = st.selectbox(
        "Data provider",
        options=["yahoo", "bloomberg", "ib"],
        index=0,
        disabled=not use_live,
    )

    st.divider()

    # ---- 4. Capital & Rebalance ---------------------------------------------
    st.header("Capital and Rebalance")

    initial_capital = st.number_input(
        "Initial capital ($)",
        min_value=1_000.0,
        max_value=100_000_000.0,
        value=1_000_000.0,
        step=100_000.0,
        format="%.0f",
    )

    rebalance = st.selectbox(
        "Rebalance frequency",
        options=["daily", "weekly", "monthly"],
        index=1,
    )

    max_position_pct = st.slider(
        "Max position %", min_value=1.0, max_value=100.0, value=10.0, step=1.0,
    )

    signal_threshold = st.slider(
        "Signal threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
    )

    st.divider()

    # ---- 4b. Position Sizing ------------------------------------------------
    st.header("Position Sizing")

    _SIZING_LABELS = ["Signal-Driven", "Fixed Dollar", "Fixed Shares", "Equal Weight"]
    _SIZING_MAP = {
        "Signal-Driven": "signal",
        "Fixed Dollar": "fixed_dollar",
        "Fixed Shares": "fixed_shares",
        "Equal Weight": "equal_weight",
    }

    sizing_label = st.radio("Sizing Mode", options=_SIZING_LABELS, index=0)
    sizing_mode = _SIZING_MAP[sizing_label]

    fixed_dollar_per_position: float | None = None
    fixed_shares_per_position: int | None = None

    if sizing_label == "Fixed Dollar":
        fixed_dollar_per_position = st.number_input(
            "Per-position dollar amount ($)",
            min_value=1_000.0,
            value=50_000.0,
            step=5_000.0,
            format="%.0f",
        )
    elif sizing_label == "Fixed Shares":
        fixed_shares_per_position = st.number_input(
            "Per-position share count",
            min_value=1,
            value=100,
            step=10,
        )

    st.divider()

    # ---- 4c. Target Volatility ----------------------------------------------
    st.header("Target Volatility")

    vol_target_enabled = st.checkbox("Enable Vol Targeting", value=False)

    vol_target_annual_pct = 10.0
    vol_lookback_days = 63
    vol_max_leverage = 3.0

    if vol_target_enabled:
        vol_target_annual_pct = st.slider(
            "Target Annual Vol %",
            min_value=1, max_value=50, value=10, step=1,
        )
        vol_lookback_days = st.slider(
            "Vol Lookback Days",
            min_value=10, max_value=252, value=63, step=1,
        )
        vol_max_leverage = st.slider(
            "Max Leverage",
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        )

    st.divider()

    # ---- 5. Fill Model ------------------------------------------------------
    st.header("Fill Model")

    fill_model = st.selectbox(
        "Fill model",
        options=["mid", "spread", "impact"],
        index=0,
    )

    if fill_model == "spread":
        spread_bps = st.slider(
            "Default spread (bps)", min_value=1.0, max_value=100.0, value=10.0, step=1.0,
        )
    else:
        spread_bps = 10.0

    if fill_model == "impact":
        impact_eta = st.slider(
            "Impact eta", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
        )
    else:
        impact_eta = 0.1

    # Execution costs (visible when fill != mid)
    if fill_model != "mid":
        st.subheader("Execution costs")
        slippage_bps = st.slider(
            "Slippage (bps)", min_value=0.0, max_value=50.0, value=3.0, step=0.5,
        )
        commission_rate = st.number_input(
            "Commission per share ($)",
            min_value=0.0,
            max_value=0.1,
            value=0.005,
            step=0.001,
            format="%.4f",
        )
        borrow_gc_rate = st.number_input(
            "Borrow GC rate",
            min_value=0.0,
            max_value=0.1,
            value=0.0025,
            step=0.0005,
            format="%.4f",
        )
    else:
        slippage_bps = 3.0
        commission_rate = 0.005
        borrow_gc_rate = 0.0025

    st.divider()

    # ---- 6. Risk Controls ---------------------------------------------------
    st.header("Risk Controls")

    risk_enabled = st.checkbox("Enable risk manager", value=False)

    if risk_enabled:
        with st.expander("Risk parameters", expanded=True):
            stop_mult = st.slider(
                "ATR stop multiplier",
                min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            )
            stop_max_loss = st.slider(
                "Max loss per position (%)",
                min_value=1.0, max_value=50.0, value=15.0, step=1.0,
            )
            sizer_max_equity = st.slider(
                "Sizer max % equity",
                min_value=1.0, max_value=50.0, value=10.0, step=1.0,
            )
            sizer_max_adv = st.slider(
                "Sizer max % ADV",
                min_value=1.0, max_value=50.0, value=10.0, step=1.0,
            )
            dd_warning = st.slider(
                "Drawdown warning (%)",
                min_value=-30.0, max_value=0.0, value=-5.0, step=0.5,
            )
            dd_halt = st.slider(
                "Drawdown halt (%)",
                min_value=-50.0, max_value=0.0, value=-15.0, step=0.5,
            )
            exp_gross = st.slider(
                "Max gross exposure (%)",
                min_value=50.0, max_value=500.0, value=200.0, step=10.0,
            )
            exp_net = st.slider(
                "Max net exposure (%)",
                min_value=5.0, max_value=200.0, value=50.0, step=5.0,
            )
            exp_single = st.slider(
                "Max single-name exposure (%)",
                min_value=1.0, max_value=100.0, value=20.0, step=1.0,
            )
    else:
        stop_mult = 3.0
        stop_max_loss = 15.0
        sizer_max_equity = 10.0
        sizer_max_adv = 10.0
        dd_warning = -5.0
        dd_halt = -15.0
        exp_gross = 200.0
        exp_net = 50.0
        exp_single = 20.0

    st.divider()

    # ---- 7. Regime ----------------------------------------------------------
    st.header("Regime Detection")

    regime_enabled = st.checkbox("Enable regime detection", value=False)
    if regime_enabled:
        regime_lookback = st.slider(
            "Regime lookback days",
            min_value=5, max_value=252, value=21, step=1,
        )
    else:
        regime_lookback = 21

    st.divider()

    # ---- RUN BUTTON ---------------------------------------------------------
    run_clicked = st.button("RUN BACKTEST", type="primary", use_container_width=True)


# ===== BUILD CONFIG AND RUN =================================================

def _parse_tickers(raw: str) -> list[str]:
    """Parse tickers from text area input (newline or comma separated)."""
    raw = raw.replace("\n", ",")
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _build_config() -> RunConfig:
    """Assemble a RunConfig from current sidebar state."""
    tickers = _parse_tickers(tickers_raw)
    if not tickers:
        raise ValueError("No tickers specified.")

    composite_weights = {"momentum": comp_mom_w, "mean_reversion": comp_mr_w}

    return RunConfig(
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        signal_threshold=signal_threshold,
        rebalance_frequency=rebalance,
        signal=SignalConfig(
            type=signal_type,
            momentum_formation_days=mom_formation,
            momentum_skip_days=mom_skip,
            momentum_zscore_clip=mom_zscore,
            mean_reversion_lookback_days=mr_lookback,
            mean_reversion_zscore_clip=mr_zscore,
            composite_weights=composite_weights,
        ),
        data=DataConfig(
            universe=tickers,
            start_date=start_date,
            end_date=end_date,
            live=use_live,
            provider=provider,
        ),
        fill=FillConfig(
            model=fill_model,
            spread_default_bps=spread_bps,
            impact_eta=impact_eta,
        ),
        execution=ExecutionConfig(
            slippage_bps=slippage_bps,
            commission_rate=commission_rate,
            borrow_gc_rate=borrow_gc_rate,
        ),
        sizing=SizingConfig(
            mode=sizing_mode,
            fixed_dollar_per_position=fixed_dollar_per_position,
            fixed_shares_per_position=fixed_shares_per_position,
        ),
        vol_target=VolTargetConfig(
            enabled=vol_target_enabled,
            target_annual_vol_pct=float(vol_target_annual_pct),
            lookback_days=vol_lookback_days,
            max_leverage=vol_max_leverage,
        ),
        risk=RiskConfig(
            enabled=risk_enabled,
            stop_multiplier=stop_mult,
            stop_max_loss_pct=stop_max_loss,
            sizer_max_pct_equity=sizer_max_equity,
            sizer_max_pct_adv=sizer_max_adv,
            drawdown_warning_pct=dd_warning,
            drawdown_halt_pct=dd_halt,
            exposure_max_gross_pct=exp_gross,
            exposure_max_net_pct=exp_net,
            exposure_max_single_name_pct=exp_single,
        ),
        regime=RegimeConfig(
            enabled=regime_enabled,
            lookback_days=regime_lookback,
        ),
        output=OutputConfig(
            tearsheet_path=None,  # generated separately below
            verbose=False,
        ),
    )


if run_clicked:
    try:
        config = _build_config()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    # Generate tearsheet to a temp file so we can embed the HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tearsheet_path = tmp.name
    config.output.tearsheet_path = tearsheet_path

    with st.spinner("Running backtest..."):
        try:
            engine, summary = run_backtest(config)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    # Read generated tearsheet HTML
    try:
        tearsheet_html = Path(tearsheet_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        tearsheet_html = None

    # Persist to session state
    st.session_state.results = summary
    st.session_state.engine = engine
    st.session_state.tearsheet_html = tearsheet_html
    st.session_state.run_config = config


# ===== MAIN AREA ===========================================================

if st.session_state.results is None:
    st.title("backtest-lab")
    st.markdown(
        "Configure your backtest in the sidebar and click **RUN BACKTEST** to begin."
    )
    st.stop()

summary = st.session_state.results
config = st.session_state.run_config

st.title("Backtest Results")
st.caption(
    f"{summary.get('signal_name', '')}  |  "
    f"{summary.get('start_date', '')} to {summary.get('end_date', '')}  |  "
    f"{summary.get('trading_days', 0)} trading days"
)

# ---------------------------------------------------------------------------
# Data Source Banner
# ---------------------------------------------------------------------------
_ds_provider = summary.get("data_provider", "")
_ds_freq = summary.get("data_frequency", "daily")
_ds_start = summary.get("data_start", "")
_ds_end = summary.get("data_end", "")
_ds_resolved = summary.get("data_tickers_resolved", 0)
_ds_requested = summary.get("data_tickers_requested", 0)
_ds_limitations = summary.get("data_limitations", "")

if _ds_provider:
    _ds_parts = [f"**{_ds_provider}** ({_ds_freq})"]
    if _ds_start and _ds_end:
        _ds_parts.append(f"{_ds_start} → {_ds_end}")
    if _ds_requested:
        _ds_parts.append(f"{_ds_resolved}/{_ds_requested} tickers resolved")
    _ds_banner = " &nbsp;|&nbsp; ".join(_ds_parts)

    _ds_limit_html = ""
    if _ds_limitations:
        _ds_limit_html = (
            f'<div style="color:#9ca3af;font-size:11px;margin-top:4px;">'
            f'{_ds_limitations}</div>'
        )

    st.markdown(
        f'<div style="background:rgba(28,31,38,0.8);border:1px solid rgba(250,250,250,0.1);'
        f'border-radius:8px;padding:10px 16px;margin-bottom:16px;">'
        f'<div style="font-size:13px;">&#128225; {_ds_banner}</div>'
        f'{_ds_limit_html}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Summary stats in 3 columns
# ---------------------------------------------------------------------------
col_ret, col_risk, col_exec = st.columns(3)

with col_ret:
    st.subheader("Returns")
    st.metric("Total return", f"{summary.get('total_return_pct', 0):.2f}%")
    st.metric("Annualized return", f"{summary.get('annualized_return_pct', 0):.2f}%")
    st.metric(
        "Final equity",
        f"${summary.get('final_equity', 0):,.0f}",
        delta=f"${summary.get('final_equity', 0) - summary.get('initial_capital', 0):,.0f}",
    )

with col_risk:
    st.subheader("Risk")
    st.metric("Sharpe ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
    st.metric("Sortino ratio", f"{summary.get('sortino_ratio', 0):.2f}")
    st.metric("Max drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%")
    st.metric("Calmar ratio", f"{summary.get('calmar_ratio', 0):.2f}")
    st.metric("Annualized vol", f"{summary.get('annualized_vol_pct', 0):.2f}%")

with col_exec:
    st.subheader("Execution")
    st.metric("Total trades", f"{summary.get('total_trades', 0):,}")
    st.metric("Total costs", f"${summary.get('total_costs', 0):,.2f}")
    st.metric("Costs as % of return", f"{summary.get('costs_as_pct_of_return', 0):.2f}%")
    st.metric("Fill model", summary.get("fill_model", "N/A"))
    st.metric("Risk manager", summary.get("risk_manager", "OFF"))
    if summary.get("risk_manager") == "ON":
        st.metric("Stops triggered", f"{summary.get('total_stops_triggered', 0)}")
        st.metric("Trades rejected", f"{summary.get('total_trades_rejected', 0)}")
    st.metric("Regime detector", summary.get("regime_detector", "OFF"))

# ---------------------------------------------------------------------------
# Trade Analytics
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Trade Analytics")

col_ta1, col_ta2, col_ta3 = st.columns(3)

with col_ta1:
    st.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
    st.metric("Slugging Pct", f"{summary.get('slugging_pct', 0):.2f}")
    st.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")

with col_ta2:
    st.metric("Avg Win", f"${summary.get('avg_win', 0):,.0f}")
    st.metric("Avg Loss", f"${summary.get('avg_loss', 0):,.0f}")
    st.metric("Avg Trade", f"${summary.get('avg_trade', 0):,.0f}")

with col_ta3:
    st.metric("Best Trade", f"${summary.get('best_trade', 0):,.0f}")
    st.metric("Worst Trade", f"${summary.get('worst_trade', 0):,.0f}")
    st.metric(
        "Win/Loss",
        f"{summary.get('total_winning_trades', 0)} / {summary.get('total_losing_trades', 0)}"
    )

# ---------------------------------------------------------------------------
# Drawdown Duration
# ---------------------------------------------------------------------------
col_dd1, col_dd2 = st.columns(2)

with col_dd1:
    st.metric("Max DD Duration", f"{summary.get('max_dd_duration_days', 0)} days")

with col_dd2:
    st.metric("Max DD Recovery", f"{summary.get('max_dd_recovery_days', 0)} days")

# ---------------------------------------------------------------------------
# Benchmark Comparison (only if data available)
# ---------------------------------------------------------------------------
if "beta" in summary and summary.get("benchmark_return_pct", 0) != 0.0:
    st.divider()
    st.subheader("Benchmark Comparison (SPY)")

    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        st.metric(
            "Benchmark Return",
            f"{summary.get('benchmark_return_pct', 0):.2f}%",
        )
        st.metric(
            "Relative Return",
            f"{summary.get('relative_return_pct', 0):+.2f}%",
            delta=f"{summary.get('relative_return_pct', 0):+.2f}%",
        )
        st.metric("Beta", f"{summary.get('beta', 0):.3f}")
        st.metric("Alpha", f"{summary.get('alpha_pct', 0):+.2f}%")

    with col_b2:
        st.metric("Info Ratio", f"{summary.get('information_ratio', 0):.3f}")
        st.metric("Tracking Error", f"{summary.get('tracking_error_pct', 0):.2f}%")
        st.metric("Correlation", f"{summary.get('benchmark_correlation', 0):.3f}")

    with col_b3:
        st.metric("Up Capture", f"{summary.get('up_capture_pct', 0):.1f}%")
        st.metric("Down Capture", f"{summary.get('down_capture_pct', 0):.1f}%")
        st.metric("R-Squared", f"{summary.get('r_squared_pct', 0):.1f}%")

# ---------------------------------------------------------------------------
# Risk Decomposition
# ---------------------------------------------------------------------------
if "risk_ann_vol_pct" in summary:
    st.divider()
    st.subheader("Risk Decomposition")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown("**Volatility**")
        st.metric("Ann. Vol", f"{summary.get('risk_ann_vol_pct', 0):.2f}%")
        st.metric("Downside Vol", f"{summary.get('risk_downside_vol_pct', 0):.2f}%")
        st.metric("21d Rolling Vol", f"{summary.get('risk_vol_21d_current_pct', 0):.2f}%")
        st.metric("63d Rolling Vol", f"{summary.get('risk_vol_63d_current_pct', 0):.2f}%")

    with col_r2:
        st.markdown("**Exposure**")
        st.metric("Avg Gross", f"{summary.get('risk_avg_gross_exposure_pct', 0):.1f}%")
        st.metric("Avg Net", f"{summary.get('risk_avg_net_exposure_pct', 0):+.1f}%")
        st.metric("Final Gross", f"{summary.get('risk_final_gross_pct', 0):.1f}%")
        st.metric("Final Net", f"{summary.get('risk_final_net_pct', 0):+.1f}%")

    with col_r3:
        st.markdown("**Concentration**")
        st.metric("HHI", f"{summary.get('risk_hhi', 0):.4f}")
        st.metric("Top 1 Name", f"{summary.get('risk_top1_pct', 0):.1f}%")
        st.metric("Top 5 Names", f"{summary.get('risk_top5_pct', 0):.1f}%")
        st.metric("Avg Positions", f"{summary.get('risk_avg_positions', 0):.1f}")

    # Beta decomposition (only if benchmark was available)
    if summary.get("risk_portfolio_beta") is not None:
        st.divider()
        st.subheader("Return Decomposition")

        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.metric("Portfolio Beta", f"{summary.get('risk_portfolio_beta', 0):.3f}")
            st.metric("Beta P&L", f"{summary.get('risk_beta_pnl_pct', 0):+.2f}%")

        with col_d2:
            st.metric("Alpha P&L", f"{summary.get('risk_alpha_pnl_pct', 0):+.2f}%")
            st.metric("Idio Ratio", f"{summary.get('risk_idio_ratio_pct', 0):.1f}%")

        with col_d3:
            if summary.get("risk_idio_vol_pct") is not None:
                st.metric("Idio Vol", f"{summary.get('risk_idio_vol_pct', 0):.2f}%")
            if summary.get("risk_idio_sharpe") is not None:
                st.metric("Idio Sharpe", f"{summary.get('risk_idio_sharpe', 0):.3f}")

# ---------------------------------------------------------------------------
# Embedded tearsheet
# ---------------------------------------------------------------------------
st.divider()

if st.session_state.tearsheet_html:
    st.subheader("Tearsheet")
    components.html(st.session_state.tearsheet_html, height=2400, scrolling=True)
else:
    st.warning("Tearsheet HTML was not generated.")

# ---------------------------------------------------------------------------
# Run directory and markdown tearsheet download
# ---------------------------------------------------------------------------
run_dir = summary.get("run_dir")
if run_dir:
    st.caption(f"Run directory: `{run_dir}`")

    md_path = summary.get("tearsheet_md")
    if md_path and Path(md_path).exists():
        md_bytes = Path(md_path).read_bytes()
        st.download_button(
            label="Download Markdown tearsheet",
            data=md_bytes,
            file_name="tearsheet.md",
            mime="text/markdown",
        )

# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Export")

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    # Export YAML config
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="w"
    ) as yaml_tmp:
        yaml_path = yaml_tmp.name
    config.to_yaml(yaml_path)
    yaml_bytes = Path(yaml_path).read_bytes()

    st.download_button(
        label="Export YAML config",
        data=yaml_bytes,
        file_name="backtest_config.yaml",
        mime="application/x-yaml",
    )

with dl_col2:
    # Download HTML tearsheet
    if st.session_state.tearsheet_html:
        st.download_button(
            label="Download HTML tearsheet",
            data=st.session_state.tearsheet_html,
            file_name="tearsheet.html",
            mime="text/html",
        )
    else:
        st.button("Download HTML tearsheet", disabled=True)


# ---------------------------------------------------------------------------
# Topic badges (GitHub-style)
# ---------------------------------------------------------------------------
st.divider()

_TOPICS = [
    "backtesting",
    "quantitative-finance",
    "long-short-equity",
    "risk-management",
    "execution-modeling",
    "market-microstructure",
    "regime-detection",
    "portfolio-construction",
    "walk-forward-validation",
    "event-driven",
    "python",
    "polars",
    "pydantic",
    "deep-learning",
    "training-data-generation",
    "financial-modeling",
]

_badge_html = " ".join(
    f'<span style="display:inline-block;background:#1f2937;color:#9ca3af;'
    f'border:1px solid #374151;border-radius:12px;padding:2px 10px;'
    f'margin:2px 2px;font-size:12px;">{t}</span>'
    for t in _TOPICS
)

st.markdown(
    f'<div style="text-align:center;padding:8px 0;">{_badge_html}</div>',
    unsafe_allow_html=True,
)
