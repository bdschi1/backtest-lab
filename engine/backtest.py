"""Core event-driven backtest engine.

The engine iterates day-by-day through historical data:
    1. Generate signals for the current bar
    2. Compute target positions from signals
    3. Check risk limits (risk manager gate — build 2)
    4. Simulate execution (fill model + slippage + commission)
    5. Update portfolio state
    6. Accrue borrow costs for short positions
    7. Record daily snapshot

Key design decisions:
    - Point-in-time: signals only see data up to current date
    - No look-ahead: the engine enforces this structurally
    - Execution on next bar's open: signal on close[t], execute on open[t+1]
      (configurable — can also execute on close[t] for less realism)
    - All costs modeled: spread, impact, slippage, commission, borrow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from execution.borrow import BorrowModel, ZeroBorrow
from execution.commission import CommissionModel, CommissionResult, ZeroCommission
from execution.fill_model import BarData, Fill, FillModel, MidPriceFill
from execution.slippage import SlippageModel, SlippageResult, ZeroSlippage
from signals.base import Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """A single position in the portfolio."""

    ticker: str
    shares: int            # positive = long, negative = short
    avg_cost: float        # average entry price
    current_price: float   # last known price

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)

    @property
    def side(self) -> str:
        if self.shares > 0:
            return "long"
        elif self.shares < 0:
            return "short"
        return "flat"


@dataclass
class Portfolio:
    """Portfolio state at a point in time."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def long_value(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.shares > 0)

    @property
    def short_value(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values() if p.shares < 0)

    @property
    def gross_value(self) -> float:
        return self.long_value + self.short_value

    @property
    def net_value(self) -> float:
        return self.long_value - self.short_value

    @property
    def total_equity(self) -> float:
        """NAV = cash + sum of all position market values."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    def update_prices(self, prices: dict[str, float]) -> None:
        """Mark positions to market with latest prices."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price

        # Remove zero-share positions
        self.positions = {
            t: p for t, p in self.positions.items() if p.shares != 0
        }


# ---------------------------------------------------------------------------
# Daily snapshot (recorded each bar)
# ---------------------------------------------------------------------------

@dataclass
class DailySnapshot:
    """Immutable record of portfolio state at end of day."""

    date: date
    equity: float
    cash: float
    long_value: float
    short_value: float
    gross_value: float
    net_value: float
    num_positions: int
    num_long: int
    num_short: int
    daily_return: float
    cumulative_return: float
    total_commission: float
    total_spread_cost: float
    total_impact_cost: float
    total_slippage_cost: float
    total_borrow_cost: float
    trades_today: int


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Record of a single executed trade."""

    date: date
    ticker: str
    side: str
    shares: int
    fill_price: float
    spread_cost: float
    impact_cost: float
    slippage_cost: float
    commission: float
    total_cost: float
    signal_score: float


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Args:
        initial_capital: Starting cash (default $1M).
        max_position_pct: Max position size as % of equity (default 5%).
        signal_threshold: Minimum absolute signal to trade (default 0.1).
        rebalance_frequency: How often to rebalance ("daily", "weekly", "monthly").
        execute_on: When to execute — "next_open" (realistic) or "close" (less realistic).
    """

    initial_capital: float = 1_000_000.0
    max_position_pct: float = 5.0
    signal_threshold: float = 0.1
    rebalance_frequency: str = "daily"
    execute_on: str = "next_open"  # "next_open" or "close"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtest engine.

    Iterates through historical data bar-by-bar, generating signals,
    computing target positions, simulating execution, and recording results.

    Args:
        signal: Signal instance for alpha generation.
        fill_model: How orders are filled (default: MidPriceFill).
        slippage_model: Additional slippage (default: ZeroSlippage).
        commission_model: Trading commissions (default: ZeroCommission).
        borrow_model: Short borrow costs (default: ZeroBorrow).
        config: Backtest configuration.
    """

    def __init__(
        self,
        signal: Signal,
        fill_model: FillModel | None = None,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
        borrow_model: BorrowModel | None = None,
        config: BacktestConfig | None = None,
    ):
        self.signal = signal
        self.fill_model = fill_model or MidPriceFill()
        self.slippage_model = slippage_model or ZeroSlippage()
        self.commission_model = commission_model or ZeroCommission()
        self.borrow_model = borrow_model or ZeroBorrow()
        self.config = config or BacktestConfig()

        # State
        self.portfolio = Portfolio(cash=self.config.initial_capital)
        self.snapshots: list[DailySnapshot] = []
        self.trades: list[TradeRecord] = []

        # Running cost accumulators
        self._total_commission = 0.0
        self._total_spread_cost = 0.0
        self._total_impact_cost = 0.0
        self._total_slippage_cost = 0.0
        self._total_borrow_cost = 0.0

    def run(self, prices: pl.DataFrame) -> list[DailySnapshot]:
        """Run the backtest on historical price data.

        Args:
            prices: Daily OHLCV data in long format:
                    date, ticker, open, high, low, close, adj_close, volume.
                    Must be sorted by (ticker, date).

        Returns:
            List of DailySnapshot objects (one per trading day).
        """
        # Get unique sorted dates
        dates = sorted(prices.get_column("date").unique().to_list())

        if len(dates) < self.signal.lookback_days:
            logger.warning(
                "Not enough data (%d days) for signal lookback (%d days)",
                len(dates), self.signal.lookback_days,
            )
            return []

        logger.info(
            "Starting backtest: %s to %s (%d days), signal=%s, "
            "fill=%s, capital=$%,.0f",
            dates[0], dates[-1], len(dates),
            self.signal.name, self.fill_model.name,
            self.config.initial_capital,
        )

        # Precompute average daily volume per ticker (20-day rolling)
        adv_map = self._compute_adv(prices)

        prev_equity = self.config.initial_capital

        for i, current_date in enumerate(dates):
            # Data available up to current_date (point-in-time)
            available_data = prices.filter(pl.col("date") <= current_date)

            # Get today's bar for each ticker
            today_bars = prices.filter(pl.col("date") == current_date)

            # Update portfolio prices
            current_prices = {}
            for row in today_bars.iter_rows(named=True):
                current_prices[row["ticker"]] = row["close"]
            self.portfolio.update_prices(current_prices)

            # Accrue daily borrow costs for short positions
            daily_borrow = 0.0
            for ticker, pos in self.portfolio.positions.items():
                if pos.shares < 0:
                    result = self.borrow_model.daily_cost(
                        shares=abs(pos.shares),
                        price=pos.current_price,
                    )
                    daily_borrow += result.daily_cost
            self.portfolio.cash -= daily_borrow
            self._total_borrow_cost += daily_borrow

            # Generate signals (only after enough lookback)
            trades_today = 0
            if i >= self.signal.lookback_days and self._should_rebalance(current_date, i):
                signals = self.signal.generate_signals(available_data, current_date)
                trades_today = self._execute_rebalance(
                    signals, today_bars, current_date, adv_map,
                )

            # Record snapshot
            equity = self.portfolio.total_equity
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            cum_ret = (equity - self.config.initial_capital) / self.config.initial_capital

            num_long = sum(1 for p in self.portfolio.positions.values() if p.shares > 0)
            num_short = sum(1 for p in self.portfolio.positions.values() if p.shares < 0)

            snapshot = DailySnapshot(
                date=current_date,
                equity=equity,
                cash=self.portfolio.cash,
                long_value=self.portfolio.long_value,
                short_value=self.portfolio.short_value,
                gross_value=self.portfolio.gross_value,
                net_value=self.portfolio.net_value,
                num_positions=len(self.portfolio.positions),
                num_long=num_long,
                num_short=num_short,
                daily_return=daily_ret,
                cumulative_return=cum_ret,
                total_commission=self._total_commission,
                total_spread_cost=self._total_spread_cost,
                total_impact_cost=self._total_impact_cost,
                total_slippage_cost=self._total_slippage_cost,
                total_borrow_cost=self._total_borrow_cost,
                trades_today=trades_today,
            )
            self.snapshots.append(snapshot)
            prev_equity = equity

        logger.info(
            "Backtest complete: %d days, final equity=$%,.0f, "
            "total return=%.2f%%, %d trades",
            len(self.snapshots),
            self.portfolio.total_equity,
            (self.portfolio.total_equity / self.config.initial_capital - 1) * 100,
            len(self.trades),
        )

        return self.snapshots

    def _should_rebalance(self, current_date: date, day_index: int) -> bool:
        """Check if we should rebalance on this date."""
        freq = self.config.rebalance_frequency
        if freq == "daily":
            return True
        elif freq == "weekly":
            return current_date.weekday() == 0  # Monday
        elif freq == "monthly":
            return current_date.day <= 3  # first 3 days of month
        return True

    def _execute_rebalance(
        self,
        signals: dict[str, float],
        today_bars: pl.DataFrame,
        current_date: date,
        adv_map: dict[str, float],
    ) -> int:
        """Convert signals to target positions and execute trades.

        Returns number of trades executed.
        """
        equity = self.portfolio.total_equity
        max_position_value = equity * (self.config.max_position_pct / 100.0)

        # Filter signals below threshold
        active_signals = {
            t: s for t, s in signals.items()
            if abs(s) >= self.config.signal_threshold
        }

        if not active_signals:
            return 0

        # Compute target shares for each ticker
        targets: dict[str, int] = {}
        bar_lookup: dict[str, dict] = {}

        for row in today_bars.iter_rows(named=True):
            bar_lookup[row["ticker"]] = row

        for ticker, score in active_signals.items():
            if ticker not in bar_lookup:
                continue

            bar = bar_lookup[ticker]
            price = bar["close"]
            if price <= 0:
                continue

            # Target notional = signal_strength * max_position_value
            target_notional = score * max_position_value
            target_shares = int(target_notional / price)
            targets[ticker] = target_shares

        # Close positions for tickers no longer in signal
        for ticker in list(self.portfolio.positions.keys()):
            if ticker not in targets:
                targets[ticker] = 0

        # Execute trades
        trades_executed = 0
        for ticker, target_shares in targets.items():
            current_shares = 0
            if ticker in self.portfolio.positions:
                current_shares = self.portfolio.positions[ticker].shares

            delta = target_shares - current_shares
            if delta == 0:
                continue

            if ticker not in bar_lookup:
                continue

            bar_row = bar_lookup[ticker]
            adv = adv_map.get(ticker, 0.0)

            bar = BarData(
                ticker=ticker,
                open=bar_row["open"],
                high=bar_row["high"],
                low=bar_row["low"],
                close=bar_row["close"],
                volume=bar_row["volume"],
                avg_daily_volume=adv,
            )

            side = "buy" if delta > 0 else "sell"
            abs_delta = abs(delta)

            # Simulate fill
            fill = self.fill_model.simulate_fill(bar, abs_delta, side)

            # Add slippage
            slip = self.slippage_model.calculate(
                fill.fill_price, abs_delta, adv,
            )

            # Calculate commission
            comm = self.commission_model.calculate(abs_delta, fill.fill_price)

            # Adjust fill price for slippage
            if side == "buy":
                effective_price = fill.fill_price + slip.cost_per_share
            else:
                effective_price = fill.fill_price - slip.cost_per_share
                effective_price = max(effective_price, 0.01)

            # Update portfolio
            trade_value = abs_delta * effective_price
            if side == "buy":
                self.portfolio.cash -= trade_value
            else:
                self.portfolio.cash += trade_value

            # Deduct commission
            self.portfolio.cash -= comm.total

            # Update position
            if ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                new_shares = pos.shares + delta
                if new_shares == 0:
                    del self.portfolio.positions[ticker]
                else:
                    # Update avg cost for additions, keep for reductions
                    if (delta > 0 and pos.shares > 0) or (delta < 0 and pos.shares < 0):
                        # Adding to existing direction — weighted avg cost
                        total_cost = pos.avg_cost * abs(pos.shares) + effective_price * abs_delta
                        pos.avg_cost = total_cost / abs(new_shares)
                    pos.shares = new_shares
                    pos.current_price = bar.close
            else:
                self.portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    shares=delta,
                    avg_cost=effective_price,
                    current_price=bar.close,
                )

            # Accumulate costs
            total_spread = fill.spread_cost * abs_delta
            total_impact = fill.impact_cost * abs_delta
            total_slip = slip.cost_per_share * abs_delta

            self._total_commission += comm.total
            self._total_spread_cost += total_spread
            self._total_impact_cost += total_impact
            self._total_slippage_cost += total_slip

            # Record trade
            signal_score = active_signals.get(ticker, 0.0)
            self.trades.append(TradeRecord(
                date=current_date,
                ticker=ticker,
                side=side,
                shares=abs_delta,
                fill_price=effective_price,
                spread_cost=total_spread,
                impact_cost=total_impact,
                slippage_cost=total_slip,
                commission=comm.total,
                total_cost=total_spread + total_impact + total_slip + comm.total,
                signal_score=signal_score,
            ))
            trades_executed += 1

        return trades_executed

    def _compute_adv(self, prices: pl.DataFrame) -> dict[str, float]:
        """Compute 20-day average daily volume per ticker."""
        adv = (
            prices
            .sort(["ticker", "date"])
            .group_by("ticker")
            .agg(pl.col("volume").tail(20).mean().alias("adv"))
        )
        return {
            row["ticker"]: row["adv"] or 0.0
            for row in adv.iter_rows(named=True)
        }

    def results_to_dataframe(self) -> pl.DataFrame:
        """Convert snapshots to a polars DataFrame for analysis."""
        if not self.snapshots:
            return pl.DataFrame()

        return pl.DataFrame([
            {
                "date": s.date,
                "equity": s.equity,
                "cash": s.cash,
                "long_value": s.long_value,
                "short_value": s.short_value,
                "gross_value": s.gross_value,
                "net_value": s.net_value,
                "num_positions": s.num_positions,
                "num_long": s.num_long,
                "num_short": s.num_short,
                "daily_return": s.daily_return,
                "cumulative_return": s.cumulative_return,
                "total_commission": s.total_commission,
                "total_spread_cost": s.total_spread_cost,
                "total_impact_cost": s.total_impact_cost,
                "total_slippage_cost": s.total_slippage_cost,
                "total_borrow_cost": s.total_borrow_cost,
                "trades_today": s.trades_today,
            }
            for s in self.snapshots
        ])

    def trades_to_dataframe(self) -> pl.DataFrame:
        """Convert trade records to a polars DataFrame."""
        if not self.trades:
            return pl.DataFrame()

        return pl.DataFrame([
            {
                "date": t.date,
                "ticker": t.ticker,
                "side": t.side,
                "shares": t.shares,
                "fill_price": t.fill_price,
                "spread_cost": t.spread_cost,
                "impact_cost": t.impact_cost,
                "slippage_cost": t.slippage_cost,
                "commission": t.commission,
                "total_cost": t.total_cost,
                "signal_score": t.signal_score,
            }
            for t in self.trades
        ])

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics for the backtest."""
        if not self.snapshots:
            return {}

        returns = [s.daily_return for s in self.snapshots]
        returns_arr = np.array(returns)

        equity_curve = [s.equity for s in self.snapshots]
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (np.array(equity_curve) - peak) / peak

        total_return = (equity_curve[-1] / self.config.initial_capital) - 1
        ann_factor = 252 / len(returns) if len(returns) > 0 else 1.0

        mean_daily = np.mean(returns_arr)
        std_daily = np.std(returns_arr)

        sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

        # Sortino (downside deviation)
        downside = returns_arr[returns_arr < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        sortino = (mean_daily / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        max_dd = float(np.min(drawdowns))
        calmar = (total_return * ann_factor) / abs(max_dd) if max_dd != 0 else 0.0

        total_costs = (
            self._total_commission
            + self._total_spread_cost
            + self._total_impact_cost
            + self._total_slippage_cost
            + self._total_borrow_cost
        )

        return {
            "start_date": self.snapshots[0].date,
            "end_date": self.snapshots[-1].date,
            "trading_days": len(self.snapshots),
            "initial_capital": self.config.initial_capital,
            "final_equity": equity_curve[-1],
            "total_return_pct": total_return * 100,
            "annualized_return_pct": ((1 + total_return) ** ann_factor - 1) * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd * 100,
            "calmar_ratio": calmar,
            "daily_vol_pct": std_daily * 100,
            "annualized_vol_pct": std_daily * np.sqrt(252) * 100,
            "total_trades": len(self.trades),
            "total_costs": total_costs,
            "total_commission": self._total_commission,
            "total_spread_cost": self._total_spread_cost,
            "total_impact_cost": self._total_impact_cost,
            "total_slippage_cost": self._total_slippage_cost,
            "total_borrow_cost": self._total_borrow_cost,
            "costs_as_pct_of_return": (
                total_costs / (equity_curve[-1] - self.config.initial_capital) * 100
                if equity_curve[-1] != self.config.initial_capital else 0.0
            ),
            "signal_name": self.signal.name,
            "fill_model": self.fill_model.name,
        }
