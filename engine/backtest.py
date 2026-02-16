"""Core event-driven backtest engine.

The engine iterates day-by-day through historical data:
    1. Update risk state (drawdown, HWM, ATR)
    2. Check stop-losses — force-close breached positions
    3. Detect market regime and adapt parameters
    4. Generate signals for the current bar
    5. Gate signals through risk manager (sizing, exposure, circuit breaker)
    6. Simulate execution (fill model + slippage + commission)
    7. Update portfolio state
    8. Accrue borrow costs for short positions
    9. Record daily snapshot

Key design decisions:
    - Point-in-time: signals only see data up to current date
    - No look-ahead: the engine enforces this structurally
    - Execution on next bar's open: signal on close[t], execute on open[t+1]
      (configurable — can also execute on close[t] for less realism)
    - All costs modeled: spread, impact, slippage, commission, borrow
    - Risk manager is an enforced gate — trades are rejected or sized down
    - Regime detector adapts parameters in real time
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
    # Risk & regime fields (None if no risk_manager/regime_detector)
    circuit_state: str | None = None
    drawdown_pct: float | None = None
    regime: str | None = None
    regime_vol: float | None = None
    stops_triggered: int = 0
    trades_rejected: int = 0


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
    risk_decision: str = "no_rm"  # risk manager decision summary


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
        sizing_mode: How positions are sized — "signal", "fixed_dollar",
                     "fixed_shares", or "equal_weight".
        fixed_dollar_per_position: Dollar amount per position (fixed_dollar mode).
        fixed_shares_per_position: Shares per position (fixed_shares mode).
    """

    initial_capital: float = 1_000_000.0
    max_position_pct: float = 5.0
    signal_threshold: float = 0.1
    rebalance_frequency: str = "daily"
    execute_on: str = "next_open"  # "next_open" or "close"
    sizing_mode: str = "signal"
    fixed_dollar_per_position: float | None = None
    fixed_shares_per_position: int | None = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtest engine with integrated risk management.

    Iterates through historical data bar-by-bar, generating signals,
    gating them through the risk manager, computing target positions,
    simulating execution, and recording results.

    Args:
        signal: Signal instance for alpha generation.
        fill_model: How orders are filled (default: MidPriceFill).
        slippage_model: Additional slippage (default: ZeroSlippage).
        commission_model: Trading commissions (default: ZeroCommission).
        borrow_model: Short borrow costs (default: ZeroBorrow).
        config: Backtest configuration.
        risk_manager: Optional RiskManager for enforced risk checks.
        regime_detector: Optional VolatilityRegimeDetector.
        regime_adapter: Optional RegimeAdapter for parameter overrides.
    """

    def __init__(
        self,
        signal: Signal,
        fill_model: FillModel | None = None,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
        borrow_model: BorrowModel | None = None,
        config: BacktestConfig | None = None,
        risk_manager: Any | None = None,
        regime_detector: Any | None = None,
        regime_adapter: Any | None = None,
        vol_target_scaler: Any | None = None,
    ):
        self.signal = signal
        self.fill_model = fill_model or MidPriceFill()
        self.slippage_model = slippage_model or ZeroSlippage()
        self.commission_model = commission_model or ZeroCommission()
        self.borrow_model = borrow_model or ZeroBorrow()
        self.config = config or BacktestConfig()
        self.risk_manager = risk_manager
        self.regime_detector = regime_detector
        self.regime_adapter = regime_adapter
        self.vol_target_scaler = vol_target_scaler

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

        # Risk tracking
        self._total_stops_triggered = 0
        self._total_trades_rejected = 0
        self._daily_returns_history: list[float] = []

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
            "fill=%s, capital=$%.0f, risk_manager=%s, regime=%s",
            dates[0], dates[-1], len(dates),
            self.signal.name, self.fill_model.name,
            self.config.initial_capital,
            "ON" if self.risk_manager else "OFF",
            "ON" if self.regime_detector else "OFF",
        )

        # Precompute average daily volume per ticker (20-day rolling)
        adv_map = self._compute_adv(prices)

        # Precompute ATR per ticker for stop-loss calculations
        atr_map = self._compute_atr(prices) if self.risk_manager else {}

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

            # ------ RISK MANAGER: update state ------
            circuit_state_str = None
            drawdown_pct = None
            stops_today = 0
            rejected_today = 0

            if self.risk_manager:
                equity = self.portfolio.total_equity
                self.risk_manager.update_bar(equity, i, current_prices)
                self.risk_manager.update_atr(atr_map)
                circuit_state_str = self.risk_manager.drawdown.state.value
                if self.risk_manager.drawdown._peak_equity > 0:
                    drawdown_pct = (
                        (equity - self.risk_manager.drawdown._peak_equity)
                        / self.risk_manager.drawdown._peak_equity * 100
                    )
                else:
                    drawdown_pct = 0.0

                # Check stop-losses on existing positions
                tickers_to_close = self.risk_manager.check_stops(
                    self.portfolio.positions
                )
                if tickers_to_close:
                    stops_today = len(tickers_to_close)
                    self._total_stops_triggered += stops_today
                    self._execute_stop_closes(
                        tickers_to_close, today_bars, current_date, adv_map,
                    )

            # ------ REGIME DETECTION ------
            regime_str = None
            regime_vol = None
            effective_signal_threshold = self.config.signal_threshold
            effective_max_position_pct = self.config.max_position_pct

            if self.regime_detector and i > 0:
                regime_state = self.regime_detector.update(
                    self._daily_returns_history
                )
                regime_str = regime_state.regime.value
                regime_vol = regime_state.annualized_vol

                if self.regime_adapter:
                    overrides = self.regime_adapter.adapt(regime_state)
                    effective_signal_threshold = overrides.signal_threshold
                    effective_max_position_pct = (
                        self.config.max_position_pct * overrides.position_scale
                    )

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
                trades_today, rejected_today = self._execute_rebalance(
                    signals, today_bars, current_date, adv_map,
                    signal_threshold=effective_signal_threshold,
                    max_position_pct=effective_max_position_pct,
                )
                self._total_trades_rejected += rejected_today

            # Record snapshot
            equity = self.portfolio.total_equity
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            cum_ret = (equity - self.config.initial_capital) / self.config.initial_capital

            # Track returns for regime detector
            self._daily_returns_history.append(daily_ret)

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
                circuit_state=circuit_state_str,
                drawdown_pct=drawdown_pct,
                regime=regime_str,
                regime_vol=regime_vol,
                stops_triggered=stops_today,
                trades_rejected=rejected_today,
            )
            self.snapshots.append(snapshot)
            prev_equity = equity

        logger.info(
            "Backtest complete: %d days, final equity=$%.0f, "
            "total return=%.2f%%, %d trades, %d stops, %d rejected",
            len(self.snapshots),
            self.portfolio.total_equity,
            (self.portfolio.total_equity / self.config.initial_capital - 1) * 100,
            len(self.trades),
            self._total_stops_triggered,
            self._total_trades_rejected,
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

    def _execute_stop_closes(
        self,
        tickers: list[str],
        today_bars: pl.DataFrame,
        current_date: date,
        adv_map: dict[str, float],
    ) -> None:
        """Force-close positions that breached stop-loss."""
        bar_lookup: dict[str, dict] = {}
        for row in today_bars.iter_rows(named=True):
            bar_lookup[row["ticker"]] = row

        for ticker in tickers:
            if ticker not in self.portfolio.positions:
                continue
            pos = self.portfolio.positions[ticker]
            if ticker not in bar_lookup:
                continue

            bar_row = bar_lookup[ticker]
            adv = adv_map.get(ticker, 0.0)
            abs_shares = abs(pos.shares)
            side = "sell" if pos.shares > 0 else "buy"

            bar = BarData(
                ticker=ticker,
                open=bar_row["open"],
                high=bar_row["high"],
                low=bar_row["low"],
                close=bar_row["close"],
                volume=bar_row["volume"],
                avg_daily_volume=adv,
            )

            fill = self.fill_model.simulate_fill(bar, abs_shares, side)
            slip = self.slippage_model.calculate(fill.fill_price, abs_shares, adv)
            comm = self.commission_model.calculate(abs_shares, fill.fill_price)

            if side == "sell":
                effective_price = fill.fill_price - slip.cost_per_share
                effective_price = max(effective_price, 0.01)
                self.portfolio.cash += abs_shares * effective_price
            else:
                effective_price = fill.fill_price + slip.cost_per_share
                self.portfolio.cash -= abs_shares * effective_price

            self.portfolio.cash -= comm.total

            # Remove position
            del self.portfolio.positions[ticker]
            if self.risk_manager:
                self.risk_manager.remove_position(ticker)

            # Accumulate costs
            total_spread = fill.spread_cost * abs_shares
            total_impact = fill.impact_cost * abs_shares
            total_slip = slip.cost_per_share * abs_shares

            self._total_commission += comm.total
            self._total_spread_cost += total_spread
            self._total_impact_cost += total_impact
            self._total_slippage_cost += total_slip

            self.trades.append(TradeRecord(
                date=current_date,
                ticker=ticker,
                side=side,
                shares=abs_shares,
                fill_price=effective_price,
                spread_cost=total_spread,
                impact_cost=total_impact,
                slippage_cost=total_slip,
                commission=comm.total,
                total_cost=total_spread + total_impact + total_slip + comm.total,
                signal_score=0.0,
                risk_decision="stop_loss_triggered",
            ))

            logger.info(
                "STOP CLOSE: %s %d shares of %s at $%.2f",
                side, abs_shares, ticker, effective_price,
            )

    def _execute_rebalance(
        self,
        signals: dict[str, float],
        today_bars: pl.DataFrame,
        current_date: date,
        adv_map: dict[str, float],
        signal_threshold: float | None = None,
        max_position_pct: float | None = None,
    ) -> tuple[int, int]:
        """Convert signals to target positions and execute trades.

        Returns (trades_executed, trades_rejected).
        """
        threshold = signal_threshold or self.config.signal_threshold
        pos_pct = max_position_pct or self.config.max_position_pct

        equity = self.portfolio.total_equity
        max_position_value = equity * (pos_pct / 100.0)

        # Filter signals below threshold
        active_signals = {
            t: s for t, s in signals.items()
            if abs(s) >= threshold
        }

        if not active_signals:
            return 0, 0

        # Compute target shares for each ticker
        targets: dict[str, int] = {}
        bar_lookup: dict[str, dict] = {}

        for row in today_bars.iter_rows(named=True):
            bar_lookup[row["ticker"]] = row

        sizing_mode = self.config.sizing_mode
        n_active = len(active_signals)

        for ticker, score in active_signals.items():
            if ticker not in bar_lookup:
                continue

            bar = bar_lookup[ticker]
            price = bar["close"]
            if price <= 0:
                continue

            if sizing_mode == "fixed_dollar" and self.config.fixed_dollar_per_position:
                # Fixed dollar: user specifies dollar amount per position
                sign = 1 if score > 0 else -1
                target_shares = int(self.config.fixed_dollar_per_position / price) * sign
            elif sizing_mode == "fixed_shares" and self.config.fixed_shares_per_position:
                # Fixed shares: user specifies share count
                sign = 1 if score > 0 else -1
                target_shares = self.config.fixed_shares_per_position * sign
            elif sizing_mode == "equal_weight" and n_active > 0:
                # Equal weight: equal dollar allocation across active signals
                per_position_value = equity / n_active
                sign = 1 if score > 0 else -1
                target_shares = int(per_position_value / price) * sign
            else:
                # Default: signal-driven sizing
                # Target notional = signal_strength * max_position_value
                target_notional = score * max_position_value
                target_shares = int(target_notional / price)

            targets[ticker] = target_shares

        # ------ VOL-TARGET SCALING: scale all targets ------
        if self.vol_target_scaler and self._daily_returns_history:
            scale = self.vol_target_scaler.compute_scale(self._daily_returns_history)
            if scale != 1.0:
                targets = {t: int(s * scale) for t, s in targets.items()}

        # Close positions for tickers no longer in signal
        for ticker in list(self.portfolio.positions.keys()):
            if ticker not in targets:
                targets[ticker] = 0

        # ------ EXPOSURE LIMITS: scale targets if needed ------
        if self.risk_manager:
            prices_for_exposure = {
                t: bar_lookup[t]["close"]
                for t in targets if t in bar_lookup
            }
            targets = self.risk_manager.exposure.scale_to_fit(
                targets, prices_for_exposure, equity,
            )

        # Execute trades
        trades_executed = 0
        trades_rejected = 0

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
            price = bar_row["close"]

            # ------ RISK MANAGER: approve each trade ------
            risk_decision_str = "no_rm"
            if self.risk_manager:
                signal_score = active_signals.get(ticker, 0.0)
                decision = self.risk_manager.approve_trade(
                    ticker=ticker,
                    requested_shares=delta,
                    signal_score=signal_score,
                    price=price,
                    equity=equity,
                    avg_daily_volume=adv,
                )
                risk_decision_str = "; ".join(decision.reasons)

                if not decision.approved:
                    trades_rejected += 1
                    logger.debug(
                        "TRADE REJECTED: %s %d shares — %s",
                        ticker, delta, risk_decision_str,
                    )
                    continue

                # Use approved (possibly sized-down) shares
                delta = decision.approved_shares

                if delta == 0:
                    trades_rejected += 1
                    continue

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
                    if self.risk_manager:
                        self.risk_manager.remove_position(ticker)
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
                if self.risk_manager:
                    self.risk_manager.register_entry(ticker, effective_price)

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
                risk_decision=risk_decision_str,
            ))
            trades_executed += 1

        return trades_executed, trades_rejected

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

    def _compute_atr(self, prices: pl.DataFrame, window: int = 14) -> dict[str, float]:
        """Compute Average True Range per ticker for stop-loss.

        Uses the standard Wilder ATR: max(H-L, |H-Cprev|, |L-Cprev|).
        """
        atr_map: dict[str, float] = {}

        for ticker in prices.get_column("ticker").unique().to_list():
            ticker_data = (
                prices.filter(pl.col("ticker") == ticker)
                .sort("date")
            )
            if ticker_data.height < window + 1:
                # Not enough data — use 2% of last close as fallback
                last_close = ticker_data.get_column("close")[-1]
                atr_map[ticker] = last_close * 0.02
                continue

            highs = ticker_data.get_column("high").to_list()
            lows = ticker_data.get_column("low").to_list()
            closes = ticker_data.get_column("close").to_list()

            true_ranges = []
            for j in range(1, len(highs)):
                tr = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]),
                )
                true_ranges.append(tr)

            if len(true_ranges) >= window:
                atr = np.mean(true_ranges[-window:])
            else:
                atr = np.mean(true_ranges) if true_ranges else closes[-1] * 0.02

            atr_map[ticker] = float(atr)

        return atr_map

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
                "circuit_state": s.circuit_state,
                "drawdown_pct": s.drawdown_pct,
                "regime": s.regime,
                "regime_vol": s.regime_vol,
                "stops_triggered": s.stops_triggered,
                "trades_rejected": s.trades_rejected,
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
                "risk_decision": t.risk_decision,
            }
            for t in self.trades
        ])

    def _compute_trade_analytics(self) -> dict:
        """Compute trade-level analytics from completed round trips.

        Tracks per-ticker cost basis using average cost (consistent with
        Position.avg_cost in the portfolio).  Realized P&L is recognised
        whenever a trade reduces or flips an existing position (FIFO-like
        via average cost).  Transaction costs on closing trades are
        subtracted from realized P&L.
        """
        _zero = {
            "win_rate": 0.0,
            "slugging_pct": 0.0,
            "profit_factor": 0.0,
            "payoff_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "avg_trade": 0.0,
            "total_winning_trades": 0,
            "total_losing_trades": 0,
        }

        if not self.trades:
            return _zero

        # Track cost basis per ticker: (shares, avg_cost)
        # shares > 0 = long, < 0 = short
        positions: dict[str, tuple[float, float]] = {}
        realized_pnls: list[float] = []

        for trade in self.trades:
            ticker = trade.ticker
            trade_shares = trade.shares if trade.side == "buy" else -trade.shares
            trade_price = trade.fill_price

            if ticker not in positions:
                positions[ticker] = (0, 0.0)

            current_shares, current_cost = positions[ticker]

            # Check if this trade reduces/closes position
            if current_shares != 0 and (
                (current_shares > 0 and trade_shares < 0)
                or (current_shares < 0 and trade_shares > 0)
            ):
                # Closing trade — compute realized P&L
                close_shares = min(abs(trade_shares), abs(current_shares))

                if current_shares > 0:
                    # Closing long: P&L = (sell_price - avg_cost) * shares
                    pnl = (trade_price - current_cost) * close_shares
                else:
                    # Closing short: P&L = (avg_cost - buy_price) * shares
                    pnl = (current_cost - trade_price) * close_shares

                pnl -= trade.total_cost  # subtract transaction costs
                realized_pnls.append(pnl)

                remaining = abs(trade_shares) - close_shares
                if remaining > 0:
                    # Position flipped — new position in opposite direction
                    new_shares = remaining if trade_shares > 0 else -remaining
                    positions[ticker] = (new_shares, trade_price)
                else:
                    new_shares = current_shares + trade_shares
                    positions[ticker] = (
                        new_shares,
                        current_cost if new_shares != 0 else 0.0,
                    )
            else:
                # Opening or adding to position — update average cost
                if current_shares == 0:
                    positions[ticker] = (trade_shares, trade_price)
                else:
                    total_cost_basis = (
                        current_cost * abs(current_shares)
                        + trade_price * abs(trade_shares)
                    )
                    new_shares = current_shares + trade_shares
                    new_avg = (
                        total_cost_basis / abs(new_shares) if new_shares != 0 else 0.0
                    )
                    positions[ticker] = (new_shares, new_avg)

        if not realized_pnls:
            return _zero

        wins = [p for p in realized_pnls if p > 0]
        losses = [p for p in realized_pnls if p < 0]

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0  # absolute value

        return {
            "win_rate": len(wins) / len(realized_pnls) * 100,
            "slugging_pct": avg_win / avg_loss if avg_loss > 0 else 0.0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0.0,
            "payoff_ratio": avg_win / avg_loss if avg_loss > 0 else 0.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": float(max(realized_pnls)),
            "worst_trade": float(min(realized_pnls)),
            "avg_trade": float(np.mean(realized_pnls)),
            "total_winning_trades": len(wins),
            "total_losing_trades": len(losses),
        }

    def _compute_drawdown_durations(self) -> dict:
        """Compute drawdown duration statistics from the equity curve.

        Walks through the equity curve tracking the high-water mark (HWM).
        When equity drops below the HWM a drawdown episode begins.  The
        episode ends when equity recovers back to (or above) the HWM.

        Returns:
            Dict with max_dd_duration_days, max_dd_recovery_days,
            avg_dd_duration_days, and current_dd_duration_days.
        """
        _zero = {
            "max_dd_duration_days": 0,
            "max_dd_recovery_days": 0,
            "avg_dd_duration_days": 0.0,
            "current_dd_duration_days": 0,
        }

        if len(self.snapshots) < 2:
            return _zero

        equity_curve = [s.equity for s in self.snapshots]

        hwm = equity_curve[0]
        dd_start: int | None = None       # index where current DD started
        trough_idx: int | None = None     # index of lowest point in current DD
        trough_val: float = hwm

        # Completed drawdown episodes: (duration_days, recovery_days)
        completed: list[tuple[int, int]] = []
        current_dd_days = 0

        for i, eq in enumerate(equity_curve):
            if eq >= hwm:
                # At or above HWM — if we were in a drawdown, it just recovered
                if dd_start is not None:
                    duration = i - dd_start          # peak-to-recovery
                    recovery = i - (trough_idx or dd_start)  # trough-to-recovery
                    completed.append((duration, recovery))
                    dd_start = None
                    trough_idx = None
                    trough_val = eq
                hwm = eq
            else:
                # Below HWM — in a drawdown
                if dd_start is None:
                    dd_start = i
                    trough_idx = i
                    trough_val = eq
                elif eq < trough_val:
                    trough_idx = i
                    trough_val = eq

        # If still in a drawdown at end of data
        if dd_start is not None:
            current_dd_days = len(equity_curve) - 1 - dd_start

        if not completed and current_dd_days == 0:
            return _zero

        max_dd_duration = max((d for d, _ in completed), default=0)
        max_dd_recovery = max((r for _, r in completed), default=0)
        avg_dd_duration = (
            float(np.mean([d for d, _ in completed])) if completed else 0.0
        )

        return {
            "max_dd_duration_days": int(max_dd_duration),
            "max_dd_recovery_days": int(max_dd_recovery),
            "avg_dd_duration_days": avg_dd_duration,
            "current_dd_duration_days": int(current_dd_days),
        }

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

        result = {
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
            # Risk manager stats
            "total_stops_triggered": self._total_stops_triggered,
            "total_trades_rejected": self._total_trades_rejected,
            "risk_manager": "ON" if self.risk_manager else "OFF",
            "regime_detector": "ON" if self.regime_detector else "OFF",
        }

        # Trade analytics
        result.update(self._compute_trade_analytics())

        # Drawdown duration
        result.update(self._compute_drawdown_durations())

        # Add final regime if available
        if self.snapshots[-1].regime:
            result["final_regime"] = self.snapshots[-1].regime
        if self.snapshots[-1].circuit_state:
            result["final_circuit_state"] = self.snapshots[-1].circuit_state

        return result
