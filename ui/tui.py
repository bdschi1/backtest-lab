"""Interactive terminal UI for backtest-lab.

Walks the user through a guided backtest setup using arrow-key menus
and text prompts. Produces a RunConfig and delegates to run_backtest().

Usage:
    backtest-lab tui
    python -m cli tui

Requires: pip install questionary (or pip install backtest-lab[tui])
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

try:
    import questionary
    from questionary import Style
except ImportError:
    print(
        "The interactive TUI requires the 'questionary' package.\n"
        "Install it with:\n\n"
        "    pip install backtest-lab[tui]\n"
        "    # or: pip install questionary\n"
    )
    sys.exit(1)

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
# Style
# ---------------------------------------------------------------------------

_STYLE = Style([
    ("qmark", "fg:ansicyan bold"),
    ("question", "bold"),
    ("answer", "fg:ansicyan"),
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("selected", "fg:ansicyan"),
])

# ---------------------------------------------------------------------------
# Preset universes
# ---------------------------------------------------------------------------

_PRESETS = {
    "Large-Cap 10": [
        "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
        "NVDA", "META", "JPM", "V", "JNJ",
    ],
    "ETF Basket": [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "TLT", "HYG", "GLD", "XLE", "XLF",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    """Print a section header."""
    width = 60
    print()
    print("-" * width)
    print(f"  {title}")
    print("-" * width)


def _float_input(prompt: str, default: str) -> float:
    """Prompt for a float value with a default."""
    raw = questionary.text(prompt, default=default, style=_STYLE).ask()
    if raw is None:
        raise KeyboardInterrupt
    return float(raw)


def _int_input(prompt: str, default: str) -> int:
    """Prompt for an integer value with a default."""
    raw = questionary.text(prompt, default=default, style=_STYLE).ask()
    if raw is None:
        raise KeyboardInterrupt
    return int(raw)


def _select(prompt: str, choices: list[str]) -> str:
    """Prompt the user to select from a list."""
    result = questionary.select(prompt, choices=choices, style=_STYLE).ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def _confirm(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no confirmation."""
    result = questionary.confirm(prompt, default=default, style=_STYLE).ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def _text(prompt: str, default: str = "") -> str:
    """Prompt for text input."""
    result = questionary.text(prompt, default=default, style=_STYLE).ask()
    if result is None:
        raise KeyboardInterrupt
    return result


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _step_signal() -> SignalConfig:
    """Step 1: Signal selection and parameter tuning."""
    _header("1/10  Signal Selection")

    sig_type = _select(
        "Signal type:",
        ["momentum", "mean_reversion", "composite"],
    )

    cfg = SignalConfig(type=sig_type)

    if sig_type in ("momentum", "composite"):
        print("\n  Momentum parameters:")
        cfg.momentum_formation_days = _int_input(
            "    Formation period (days):", str(cfg.momentum_formation_days),
        )
        cfg.momentum_skip_days = _int_input(
            "    Skip period (days):", str(cfg.momentum_skip_days),
        )
        cfg.momentum_zscore_clip = _float_input(
            "    Z-score clip:", str(cfg.momentum_zscore_clip),
        )

    if sig_type in ("mean_reversion", "composite"):
        print("\n  Mean-reversion parameters:")
        cfg.mean_reversion_lookback_days = _int_input(
            "    Lookback period (days):", str(cfg.mean_reversion_lookback_days),
        )
        cfg.mean_reversion_zscore_clip = _float_input(
            "    Z-score clip:", str(cfg.mean_reversion_zscore_clip),
        )

    if sig_type == "composite":
        print("\n  Composite weights (must sum to 1.0):")
        mom_w = _float_input("    Momentum weight:", "0.6")
        mr_w = _float_input("    Mean-reversion weight:", "0.4")
        cfg.composite_weights = {"momentum": mom_w, "mean_reversion": mr_w}

    return cfg


def _step_data() -> DataConfig:
    """Step 2: Universe and data source."""
    _header("2/10  Universe & Data")

    print(
        "  Note: Data is provided by yfinance (daily OHLCV only). No intraday,\n"
        "  no real-time, no bid/ask. Any NYSE/NASDAQ/global ticker is supported.\n"
    )

    preset = _select(
        "Ticker universe:",
        ["Large-Cap 10", "ETF Basket", "Custom"],
    )

    if preset == "Custom":
        raw = _text(
            "Enter tickers (comma-separated):",
            default="AAPL,MSFT,GOOG",
        )
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    else:
        tickers = _PRESETS[preset]
        print(f"  -> {', '.join(tickers)}")

    data_mode = _select(
        "Data source:",
        ["Synthetic (no API needed)", "Live market data"],
    )
    live = data_mode.startswith("Live")

    provider = "yahoo"
    if live:
        provider = _select("Data provider:", ["yahoo", "bloomberg", "ib"])

    start_str = _text(
        "Start date (YYYY-MM-DD, blank = 3 years ago):",
        default="",
    )
    end_str = _text(
        "End date (YYYY-MM-DD, blank = today):",
        default="",
    )

    start_date = date.fromisoformat(start_str) if start_str.strip() else None
    end_date = date.fromisoformat(end_str) if end_str.strip() else None

    return DataConfig(
        universe=tickers,
        start_date=start_date,
        end_date=end_date,
        live=live,
        provider=provider,
    )


def _step_execution() -> tuple[FillConfig, ExecutionConfig]:
    """Step 3: Execution tier (simplified)."""
    _header("3/10  Execution Tier")

    print("  Select the level of execution realism:\n")
    print("    Prototyping  - Mid-price fills, no transaction costs")
    print("    Realistic    - Spread-aware fills, 3 bps slippage, $0.005/share")
    print("    Production   - Market-impact fills, full cost stack\n")

    tier = _select(
        "Execution tier:",
        ["Prototyping", "Realistic", "Production"],
    )

    if tier == "Prototyping":
        fill = FillConfig(model="mid")
        exe = ExecutionConfig(slippage_bps=0.0, commission_rate=0.0, borrow_gc_rate=0.0)
    elif tier == "Realistic":
        fill = FillConfig(model="spread", spread_default_bps=10.0)
        exe = ExecutionConfig(slippage_bps=3.0, commission_rate=0.005, borrow_gc_rate=0.0025)
    else:
        fill = FillConfig(model="impact", impact_eta=0.1)
        exe = ExecutionConfig(slippage_bps=5.0, commission_rate=0.005, borrow_gc_rate=0.005)

    return fill, exe


def _step_capital() -> tuple[float, str, float, float]:
    """Step 4: Capital, rebalance frequency, position sizing."""
    _header("4/10  Capital & Rebalance")

    capital = _float_input("Initial capital ($):", "1000000")
    rebalance = _select(
        "Rebalance frequency:",
        ["daily", "weekly", "monthly"],
    )
    max_pos = _float_input("Max position (% of equity):", "10.0")
    threshold = _float_input("Signal threshold (0.0 - 1.0):", "0.1")

    return capital, rebalance, max_pos, threshold


def _step_sizing() -> SizingConfig:
    """Step 5: Position sizing mode."""
    _header("5/10  Position Sizing")

    mode_label = _select(
        "How do you want to size positions?",
        [
            "Signal-Driven (default)",
            "Fixed Dollar per Position",
            "Fixed Shares per Position",
            "Equal Weight",
        ],
    )

    mode_map = {
        "Signal-Driven (default)": "signal",
        "Fixed Dollar per Position": "fixed_dollar",
        "Fixed Shares per Position": "fixed_shares",
        "Equal Weight": "equal_weight",
    }
    mode = mode_map[mode_label]

    fixed_dollar: float | None = None
    fixed_shares: int | None = None

    if mode == "fixed_dollar":
        fixed_dollar = _float_input("  Dollar amount per position ($):", "50000")
    elif mode == "fixed_shares":
        fixed_shares = _int_input("  Shares per position:", "100")

    return SizingConfig(
        mode=mode,
        fixed_dollar_per_position=fixed_dollar,
        fixed_shares_per_position=fixed_shares,
    )


def _step_vol_target() -> VolTargetConfig:
    """Step 6: Volatility targeting."""
    _header("6/10  Volatility Target")

    enabled = _confirm("Target a specific portfolio volatility?", default=False)
    if not enabled:
        return VolTargetConfig(enabled=False)

    target_vol = _float_input("  Target annual vol % (1-50):", "10")
    lookback = _int_input("  Vol lookback days (10-252):", "63")

    return VolTargetConfig(
        enabled=True,
        target_annual_vol_pct=target_vol,
        lookback_days=lookback,
    )


def _step_risk() -> RiskConfig:
    """Step 7: Risk controls (renumbered from original step 5)."""
    _header("7/10  Risk Controls")

    enabled = _confirm("Enable risk manager?", default=False)
    if not enabled:
        return RiskConfig(enabled=False)

    print("\n  Risk parameters (press Enter to accept defaults):")

    stop_mult = _float_input("    ATR stop multiplier:", "3.0")
    stop_loss = _float_input("    Max stop-loss (%):", "15.0")
    max_equity = _float_input("    Max position (% equity):", "10.0")
    max_adv = _float_input("    Max position (% ADV):", "10.0")
    dd_warn = _float_input("    Drawdown warning (%):", "-5.0")
    dd_halt = _float_input("    Drawdown halt (%):", "-15.0")
    max_gross = _float_input("    Max gross exposure (%):", "200.0")
    max_net = _float_input("    Max net exposure (%):", "50.0")
    max_single = _float_input("    Max single-name (%):", "20.0")

    return RiskConfig(
        enabled=True,
        stop_multiplier=stop_mult,
        stop_max_loss_pct=stop_loss,
        sizer_max_pct_equity=max_equity,
        sizer_max_pct_adv=max_adv,
        drawdown_warning_pct=dd_warn,
        drawdown_halt_pct=dd_halt,
        exposure_max_gross_pct=max_gross,
        exposure_max_net_pct=max_net,
        exposure_max_single_name_pct=max_single,
    )


def _step_regime() -> RegimeConfig:
    """Step 8: Regime detection."""
    _header("8/10  Regime Detection")

    enabled = _confirm("Enable volatility regime detection?", default=False)
    if not enabled:
        return RegimeConfig(enabled=False)

    lookback = _int_input("  Lookback period (days):", "21")

    return RegimeConfig(enabled=True, lookback_days=lookback)


def _step_output() -> tuple[OutputConfig, str | None]:
    """Step 9: Output options.

    Returns:
        (OutputConfig, yaml_save_path or None)
    """
    _header("9/10  Output")

    tearsheet = _confirm("Generate HTML tear sheet?", default=False)
    tearsheet_path = None
    if tearsheet:
        tearsheet_path = _text("  Tear sheet path:", default="tearsheet.html")

    save_yaml = _confirm("Save this config as YAML?", default=False)
    yaml_path = None
    if save_yaml:
        yaml_path = _text("  YAML save path:", default="backtest_config.yaml")

    output_cfg = OutputConfig(tearsheet_path=tearsheet_path)

    return output_cfg, yaml_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(config: RunConfig, yaml_path: str | None) -> None:
    """Print a formatted summary of the configuration before running."""
    _header("10/10  Configuration Summary")

    end_dt = config.data.end_date or date.today()
    start_dt = config.data.start_date or (end_dt - timedelta(days=3 * 365))
    data_src = "live" if config.data.live else "synthetic"
    tickers_str = ", ".join(config.data.universe[:5])
    if len(config.data.universe) > 5:
        tickers_str += f" (+{len(config.data.universe) - 5} more)"

    # Sizing summary
    sizing_mode = config.sizing.mode
    if sizing_mode == "fixed_dollar" and config.sizing.fixed_dollar_per_position:
        sizing_str = f"Fixed ${config.sizing.fixed_dollar_per_position:,.0f}/position"
    elif sizing_mode == "fixed_shares" and config.sizing.fixed_shares_per_position:
        sizing_str = f"Fixed {config.sizing.fixed_shares_per_position} shares/position"
    elif sizing_mode == "equal_weight":
        sizing_str = "Equal weight"
    else:
        sizing_str = "Signal-driven"

    # Vol target summary
    if config.vol_target.enabled:
        vol_str = (
            f"{config.vol_target.target_annual_vol_pct:.0f}% annual "
            f"({config.vol_target.lookback_days}d lookback)"
        )
    else:
        vol_str = "OFF"

    rows = [
        ("Signal", config.signal.type),
        ("Universe", f"{len(config.data.universe)} tickers: {tickers_str}"),
        ("Data source", f"{data_src} ({config.data.provider})"),
        ("Period", f"{start_dt} to {end_dt}"),
        ("Capital", f"${config.initial_capital:,.0f}"),
        ("Rebalance", config.rebalance_frequency),
        ("Max position", f"{config.max_position_pct:.1f}%"),
        ("Signal threshold", f"{config.signal_threshold}"),
        ("Sizing", sizing_str),
        ("Vol target", vol_str),
        ("Fill model", config.fill.model),
        ("Slippage", f"{config.execution.slippage_bps:.1f} bps"),
        ("Commission", f"${config.execution.commission_rate}/share"),
        ("Risk manager", "ON" if config.risk.enabled else "OFF"),
        ("Regime detection", "ON" if config.regime.enabled else "OFF"),
        ("Tear sheet", config.output.tearsheet_path or "none"),
        ("Save config", yaml_path or "none"),
    ]

    label_width = max(len(r[0]) for r in rows) + 2
    for label, value in rows:
        print(f"  {label:<{label_width}} {value}")
    print()


def _print_results(config: RunConfig, summary: dict) -> None:
    """Print backtest results in the same format as the CLI."""
    print()
    print("=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    if not summary:
        print("  No summary data available.")
        return

    print(f"  Total Return:      {summary.get('total_return_pct', 0):+.2f}%")
    print(f"  Sharpe Ratio:      {summary.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:     {summary.get('sortino_ratio', 0):.2f}")
    print(f"  Max Drawdown:      {summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Calmar Ratio:      {summary.get('calmar_ratio', 0):.2f}")
    print(f"  Ann. Vol:          {summary.get('annualized_vol_pct', 0):.2f}%")
    print(f"  Total Trades:      {summary.get('total_trades', 0)}")
    print(f"  Total Costs:       ${summary.get('total_costs', 0):,.2f}")

    if config.risk.enabled:
        print(f"  Stops Triggered:   {summary.get('total_stops_triggered', 0)}")
        print(f"  Trades Rejected:   {summary.get('total_trades_rejected', 0)}")

    if config.regime.enabled and "final_regime" in summary:
        print(f"  Final Regime:      {summary['final_regime']}")

    if config.output.tearsheet_path:
        print(f"\n  Tear sheet saved:  {config.output.tearsheet_path}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_tui() -> None:
    """Interactive terminal UI entry point.

    Walks through each configuration step, displays a summary,
    and runs the backtest on confirmation.
    """
    try:
        print()
        print("=" * 60)
        print("  backtest-lab  --  Interactive Setup")
        print("=" * 60)
        print("  Use arrow keys to navigate, Enter to select.")
        print("  Press Ctrl+C at any time to cancel.")

        # Step 1: Signal
        signal_cfg = _step_signal()

        # Step 2: Universe & Data
        data_cfg = _step_data()

        # Step 3: Execution Tier
        fill_cfg, exec_cfg = _step_execution()

        # Step 4: Capital & Rebalance
        capital, rebalance, max_pos, threshold = _step_capital()

        # Step 5: Position Sizing
        sizing_cfg = _step_sizing()

        # Step 6: Volatility Target
        vol_target_cfg = _step_vol_target()

        # Step 7: Risk Controls
        risk_cfg = _step_risk()

        # Step 8: Regime Detection
        regime_cfg = _step_regime()

        # Step 9: Output
        output_cfg, yaml_path = _step_output()

        # Build RunConfig
        config = RunConfig(
            initial_capital=capital,
            max_position_pct=max_pos,
            signal_threshold=threshold,
            rebalance_frequency=rebalance,
            signal=signal_cfg,
            data=data_cfg,
            fill=fill_cfg,
            execution=exec_cfg,
            sizing=sizing_cfg,
            vol_target=vol_target_cfg,
            risk=risk_cfg,
            regime=regime_cfg,
            output=output_cfg,
        )

        # Step 10: Summary and confirmation
        _print_summary(config, yaml_path)

        proceed = _confirm("Run this backtest?", default=True)
        if not proceed:
            print("\n  Cancelled. No backtest was run.")
            return

        # Save YAML if requested
        if yaml_path:
            config.to_yaml(yaml_path)
            print(f"\n  Config saved to: {yaml_path}")

        # Run
        n_tickers = len(config.data.universe)
        data_src = "live" if config.data.live else "synthetic"
        print(f"\n  Data: {n_tickers} tickers ({data_src})")
        print(f"  Running: {config.signal.type} | {config.fill.model} | "
              f"risk={'ON' if config.risk.enabled else 'OFF'} | "
              f"regime={'ON' if config.regime.enabled else 'OFF'}")
        print("-" * 60)

        engine, summary = run_backtest(config)
        _print_results(config, summary)

    except KeyboardInterrupt:
        print("\n\n  Interrupted. Exiting.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  Error: {exc}")
        sys.exit(1)
