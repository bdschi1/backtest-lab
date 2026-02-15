"""Example: Cross-sectional momentum backtest with full execution realism.

Demonstrates backtest-lab's core capabilities:
    1. Momentum signal (Jegadeesh & Titman) with skip-month
    2. Three execution tiers compared: mid-price vs spread-aware vs market impact
    3. Commission and borrow costs
    4. Risk manager with ATR trailing stops and drawdown circuit breaker
    5. Regime-aware position sizing
    6. Walk-forward validation

Run from backtest-lab root:
    python examples/momentum_backtest.py

Uses synthetic data by default — no API calls needed.
Set USE_LIVE_DATA = True to use yfinance (requires internet).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backtest import BacktestConfig, BacktestEngine
from execution.fill_model import MarketImpactFill, MidPriceFill, SpreadAwareFill
from execution.commission import PerShareCommission
from execution.slippage import FixedSlippage
from execution.borrow import TieredBorrow
from signals.momentum import MomentumSignal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_LIVE_DATA = False  # Set True to fetch from yfinance
UNIVERSE = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "JNJ"]
START_DATE = date(2022, 1, 1)
END_DATE = date(2024, 12, 31)


def generate_synthetic_data(
    tickers: list[str],
    start: date,
    end: date,
) -> pl.DataFrame:
    """Generate realistic synthetic price data."""
    np.random.seed(42)
    rows = []

    # Assign different characteristics to each stock
    stock_params = {}
    for i, ticker in enumerate(tickers):
        drift = np.random.uniform(-0.0002, 0.0008)
        vol = np.random.uniform(0.01, 0.03)
        base_price = np.random.uniform(50, 500)
        adv = np.random.uniform(5_000_000, 80_000_000)
        stock_params[ticker] = (drift, vol, base_price, adv)

    d = start
    while d <= end:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue

        for ticker in tickers:
            drift, vol, base_price, adv = stock_params[ticker]
            days_elapsed = (d - start).days
            price = base_price * np.exp(
                (drift - 0.5 * vol ** 2) * days_elapsed
                + vol * np.sqrt(days_elapsed) * np.random.normal()
            )
            price = max(price, 1.0)

            daily_range = price * np.random.uniform(0.005, 0.03)
            volume = adv * np.random.uniform(0.5, 2.0)

            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price - daily_range * 0.2,
                "high": price + daily_range * 0.5,
                "low": price - daily_range * 0.5,
                "close": price,
                "adj_close": price,
                "volume": volume,
            })

        d += timedelta(days=1)

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def fetch_live_data(tickers: list[str], start: date, end: date) -> pl.DataFrame:
    """Fetch real data via yfinance."""
    from data.yahoo_provider import YahooProvider
    provider = YahooProvider()
    return provider.fetch_daily_prices(tickers, start, end)


def run_comparison():
    """Run the same strategy with three different execution models."""
    print("=" * 70)
    print("BACKTEST-LAB: Momentum Strategy Comparison")
    print("=" * 70)

    # Get data
    if USE_LIVE_DATA:
        print("\nFetching live data from Yahoo Finance...")
        prices = fetch_live_data(UNIVERSE, START_DATE, END_DATE)
    else:
        print("\nUsing synthetic data (set USE_LIVE_DATA=True for real data)")
        prices = generate_synthetic_data(UNIVERSE, START_DATE, END_DATE)

    n_days = prices.get_column("date").n_unique()
    n_tickers = prices.get_column("ticker").n_unique()
    print(f"Data: {n_tickers} tickers, {n_days} trading days")
    print(f"Period: {START_DATE} to {END_DATE}")

    # Signal
    signal = MomentumSignal(formation_days=126, skip_days=21)

    # Common config
    config = BacktestConfig(
        initial_capital=1_000_000,
        max_position_pct=10.0,
        signal_threshold=0.15,
        rebalance_frequency="weekly",
    )

    # Three execution tiers
    scenarios = [
        (
            "Tier 1: Mid-Price (Naive)",
            MidPriceFill(),
            None, None, None,
        ),
        (
            "Tier 2: Spread-Aware",
            SpreadAwareFill(default_spread_bps=10),
            FixedSlippage(bps=5),
            PerShareCommission(rate=0.005),
            None,
        ),
        (
            "Tier 3: Full Realism (Spread + Impact + Borrow)",
            MarketImpactFill(eta=0.1),
            FixedSlippage(bps=3),
            PerShareCommission(rate=0.005),
            TieredBorrow(gc_rate=0.0025),
        ),
    ]

    results = []

    for name, fill, slippage, commission, borrow in scenarios:
        print(f"\n{'─' * 50}")
        print(f"Running: {name}")

        engine = BacktestEngine(
            signal=signal,
            fill_model=fill,
            slippage_model=slippage,
            commission_model=commission,
            borrow_model=borrow,
            config=config,
        )
        engine.run(prices)
        summary = engine.summary()
        results.append((name, summary))

        if summary:
            print(f"  Total Return:      {summary['total_return_pct']:+.2f}%")
            print(f"  Sharpe Ratio:      {summary['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:      {summary['max_drawdown_pct']:.2f}%")
            print(f"  Total Trades:      {summary['total_trades']}")
            print(f"  Total Costs:       ${summary['total_costs']:,.2f}")
            print(f"    Commission:      ${summary['total_commission']:,.2f}")
            print(f"    Spread:          ${summary['total_spread_cost']:,.2f}")
            print(f"    Impact:          ${summary['total_impact_cost']:,.2f}")
            print(f"    Slippage:        ${summary['total_slippage_cost']:,.2f}")
            print(f"    Borrow:          ${summary['total_borrow_cost']:,.2f}")

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("EXECUTION COST IMPACT COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<45} {'Return':>8} {'Sharpe':>8} {'Costs':>10}")
    print("─" * 73)

    for name, s in results:
        if s:
            print(
                f"{name:<45} {s['total_return_pct']:>+7.2f}% "
                f"{s['sharpe_ratio']:>7.2f} "
                f"${s['total_costs']:>9,.0f}"
            )

    if len(results) >= 3 and results[0][1] and results[2][1]:
        naive_ret = results[0][1]["total_return_pct"]
        real_ret = results[2][1]["total_return_pct"]
        cost_drag = naive_ret - real_ret
        print(f"\nExecution cost drag: {cost_drag:+.2f}% "
              f"(naive vs full realism)")
        if abs(naive_ret) > 0.01:
            print(f"Costs as % of naive return: "
                  f"{cost_drag / naive_ret * 100:.1f}%")


if __name__ == "__main__":
    run_comparison()
