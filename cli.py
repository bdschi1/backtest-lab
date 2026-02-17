"""CLI entry point for backtest-lab.

Usage:
    python -m cli run --signal momentum --universe AAPL,MSFT,GOOG --start 2022-01-01
    python -m cli run --config examples/configs/ls_with_risk.yaml
    python -m cli run --signal momentum --risk --save-config my_backtest.yaml
    python -m cli tui
    python -m cli providers
    python -m cli example

Or if installed via pip:
    backtest-lab run --signal momentum --universe AAPL,MSFT,GOOG
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path



def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="backtest-lab",
        description="Event-driven backtesting engine with execution realism.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="Run a backtest")
    run_parser.add_argument(
        "--signal", choices=["momentum", "mean_reversion", "composite"],
        default="momentum", help="Signal type (default: momentum)",
    )
    run_parser.add_argument(
        "--universe", type=str,
        default="AAPL,MSFT,GOOG,AMZN,TSLA,NVDA,META,JPM,V,JNJ",
        help="Comma-separated tickers (default: 10 large caps)",
    )
    run_parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (default: 3 years ago)",
    )
    run_parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    run_parser.add_argument(
        "--capital", type=float, default=1_000_000,
        help="Initial capital (default: $1,000,000)",
    )
    run_parser.add_argument(
        "--live", action="store_true",
        help="Use live yfinance data (default: synthetic)",
    )
    run_parser.add_argument(
        "--provider", choices=["yahoo", "bloomberg", "ib"],
        default="yahoo", help="Data provider (default: yahoo)",
    )
    run_parser.add_argument(
        "--fill", choices=["mid", "spread", "impact"],
        default="mid", help="Fill model (default: mid)",
    )
    run_parser.add_argument(
        "--risk", action="store_true",
        help="Enable risk manager",
    )
    run_parser.add_argument(
        "--regime", action="store_true",
        help="Enable regime detection",
    )
    run_parser.add_argument(
        "--tearsheet", type=str, default=None,
        help="Generate HTML tear sheet at path",
    )
    run_parser.add_argument(
        "--rebalance", choices=["daily", "weekly", "monthly"],
        default="weekly", help="Rebalance frequency (default: weekly)",
    )
    run_parser.add_argument(
        "--max-position-pct", type=float, default=10.0,
        help="Max position size %% of equity (default: 10)",
    )
    run_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides CLI flags)",
    )
    run_parser.add_argument(
        "--save-config", type=str, default=None,
        help="Save the current run config to a YAML file",
    )
    run_parser.add_argument(
        "--sizing-mode",
        choices=["signal", "fixed_dollar", "fixed_shares", "equal_weight"],
        default="signal",
        help="Position sizing mode (default: signal)",
    )
    run_parser.add_argument(
        "--fixed-dollar", type=float, default=None,
        help="Dollar amount per position (requires --sizing-mode fixed_dollar)",
    )
    run_parser.add_argument(
        "--fixed-shares", type=int, default=None,
        help="Share count per position (requires --sizing-mode fixed_shares)",
    )
    run_parser.add_argument(
        "--vol-target", type=float, default=None,
        help="Target annual vol %%; enables vol targeting when provided",
    )
    run_parser.add_argument(
        "--vol-lookback", type=int, default=63,
        help="Lookback days for vol targeting (default: 63)",
    )
    run_parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Base results directory (default: results)",
    )

    # ---- tui ----
    subparsers.add_parser("tui", help="Interactive terminal setup")

    # ---- providers ----
    subparsers.add_parser("providers", help="List available data providers")

    # ---- example ----
    subparsers.add_parser("example", help="Run the example momentum backtest")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "tui":
        _cmd_tui()
    elif args.command == "providers":
        _cmd_providers()
    elif args.command == "example":
        _cmd_example()
    else:
        parser.print_help()


def _cmd_run(args):
    """Run a backtest using the shared runner."""
    from btconfig.run_config import RunConfig
    from btconfig.runner import run_backtest

    # Build RunConfig -- from YAML or CLI args
    if args.config:
        rc = RunConfig.from_yaml(args.config)
    else:
        rc = RunConfig.from_cli_args(args)

    # Optionally save the config
    if args.save_config:
        rc.to_yaml(args.save_config)
        print(f"Config saved to: {args.save_config}")

    # Print run header
    n_tickers = len(rc.data.universe)
    end = rc.data.end_date or date.today()
    start = rc.data.start_date or (end - timedelta(days=3 * 365))
    data_src = "live" if rc.data.live else "synthetic"
    print(f"\nData: {n_tickers} tickers ({data_src})")
    print(f"Period: {start} to {end}")
    print(f"\nRunning: {rc.signal.type} | {rc.fill.model} | "
          f"risk={'ON' if rc.risk.enabled else 'OFF'} | "
          f"regime={'ON' if rc.regime.enabled else 'OFF'}")
    print("\u2500" * 60)

    # Run
    engine, s = run_backtest(rc)

    # Print summary
    if s:
        print(f"  Total Return:      {s['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio:      {s['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:     {s['sortino_ratio']:.2f}")
        print(f"  Max Drawdown:      {s['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio:      {s['calmar_ratio']:.2f}")
        print(f"  Ann. Vol:          {s['annualized_vol_pct']:.2f}%")
        print(f"  Total Trades:      {s['total_trades']}")
        print(f"  Total Costs:       ${s['total_costs']:,.2f}")
        if rc.risk.enabled:
            print(f"  Stops Triggered:   {s['total_stops_triggered']}")
            print(f"  Trades Rejected:   {s['total_trades_rejected']}")
        if rc.regime.enabled and "final_regime" in s:
            print(f"  Final Regime:      {s['final_regime']}")

    if rc.output.tearsheet_path:
        print(f"\nTear sheet saved: {rc.output.tearsheet_path}")


def _cmd_tui():
    """Launch the interactive TUI."""
    try:
        from ui.tui import run_tui
    except ImportError:
        print(
            "TUI requires questionary. Install it with:\n"
            "  pip install backtest-lab[tui]\n"
            "  # or: pip install questionary"
        )
        sys.exit(1)
    run_tui()


def _cmd_providers():
    """List available data providers."""
    print("Available data providers:")
    print("  yahoo       \u2014 Yahoo Finance via yfinance (always available)")

    try:
        from data.bloomberg_provider import is_available as bbg_available
        status = "AVAILABLE" if bbg_available() else "not installed (pip install backtest-lab[bloomberg])"
    except ImportError:
        status = "not installed"
    print(f"  bloomberg   \u2014 Bloomberg Terminal via blpapi ({status})")

    try:
        from data.ib_provider import is_available as ib_available
        status = "AVAILABLE" if ib_available() else "not installed (pip install backtest-lab[ibkr])"
    except ImportError:
        status = "not installed"
    print(f"  ib          \u2014 Interactive Brokers via ib_insync ({status})")


def _cmd_example():
    """Run the example momentum backtest."""
    import subprocess
    example_path = Path(__file__).parent / "examples" / "momentum_backtest.py"
    if example_path.exists():
        subprocess.run([sys.executable, str(example_path)])
    else:
        print(f"Example not found at {example_path}")


# Legacy data helpers kept for backward compatibility with tests
def _fetch_live(tickers, start, end, provider_name):
    """Fetch live data -- delegates to config.runner."""
    from btconfig.runner import _fetch_live as _runner_fetch
    return _runner_fetch(tickers, start, end, provider_name)


def _generate_synthetic(tickers, start, end):
    """Generate synthetic data -- delegates to config.runner."""
    from btconfig.runner import _generate_synthetic as _runner_synth
    return _runner_synth(tickers, start, end)


if __name__ == "__main__":
    main()
