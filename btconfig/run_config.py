"""Pydantic configuration model — single source of truth for all interfaces.

Every backtest run is fully described by a RunConfig. The CLI, YAML files,
TUI menus, and Streamlit dashboard all produce a RunConfig instance, which
is then passed to run_backtest() for execution.

Design notes:
    - Nested sub-models keep YAML configs clean and self-documenting.
    - All defaults match the current CLI defaults (backward-compatible).
    - Field validators enforce sensible ranges.
    - from_cli_args() maps the flat argparse namespace to the nested model.
    - from_yaml() / to_yaml() for file-based configs.
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SignalConfig(BaseModel):
    """Signal selection and tuning."""

    type: Literal["momentum", "mean_reversion", "composite"] = "momentum"

    # Momentum
    momentum_formation_days: int = Field(126, ge=20, le=504)
    momentum_skip_days: int = Field(21, ge=0, le=63)
    momentum_zscore_clip: float = Field(3.0, ge=1.0, le=5.0)

    # Mean-reversion
    mean_reversion_lookback_days: int = Field(21, ge=5, le=252)
    mean_reversion_zscore_clip: float = Field(3.0, ge=1.0, le=5.0)

    # Composite
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {"momentum": 0.6, "mean_reversion": 0.4}
    )


class DataConfig(BaseModel):
    """Data source configuration.

    yfinance (the default provider) delivers daily OHLCV data for any
    ticker on NYSE, NASDAQ, or major global exchanges. See
    YFINANCE_LIMITATIONS for what it does NOT support.
    """

    YFINANCE_LIMITATIONS: ClassVar[str] = (
        "yfinance provides daily OHLCV data only. No intraday, no real-time, "
        "no bid/ask. Data may be delayed 15-20 min. Some tickers may be "
        "unavailable. Adjusted close accounts for splits and dividends. "
        "Any ticker listed on NYSE, NASDAQ, or major global exchanges is supported."
    )

    universe: list[str] = Field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
            "NVDA", "META", "JPM", "V", "JNJ",
        ]
    )
    start_date: date | None = None   # None -> 3 years ago
    end_date: date | None = None     # None -> today
    live: bool = True                # True = use yfinance, False = synthetic
    provider: Literal["yahoo", "bloomberg", "ib"] = "yahoo"

    @field_validator("universe", mode="before")
    @classmethod
    def _parse_universe(cls, v):
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return [t.strip().upper() for t in v.split(",") if t.strip()]
        return [t.upper() for t in v]


class FillConfig(BaseModel):
    """Fill model selection and parameters."""

    model: Literal["mid", "spread", "impact"] = "mid"
    spread_default_bps: float = Field(10.0, ge=1.0, le=100.0)
    impact_eta: float = Field(0.1, ge=0.01, le=1.0)


class ExecutionConfig(BaseModel):
    """Slippage, commission, and borrow cost parameters."""

    slippage_bps: float = Field(3.0, ge=0.0, le=50.0)
    commission_rate: float = Field(0.005, ge=0.0)
    borrow_gc_rate: float = Field(0.0025, ge=0.0)


class SizingConfig(BaseModel):
    """Position sizing configuration.

    mode determines how positions are sized:
      - "signal": signal_score x max_pct_equity x equity (current default)
      - "fixed_dollar": user specifies dollar amount per position
      - "fixed_shares": user specifies share count per position
      - "equal_weight": equal dollar allocation across all positions

    When fixed_dollar is set, shares auto-compute from current price.
    When fixed_shares is set, dollar value auto-computes from current price.
    """

    mode: Literal["signal", "fixed_dollar", "fixed_shares", "equal_weight"] = "signal"
    fixed_dollar_per_position: float | None = None   # e.g. $50,000
    fixed_shares_per_position: int | None = None      # e.g. 100


class VolTargetConfig(BaseModel):
    """Target volatility scaling.

    When enabled, portfolio positions are scaled so that the realized
    portfolio volatility tracks the target. Uses trailing realized vol
    (lookback_days) and rescales gross exposure each rebalance.

    Reference: Moskowitz, Ooi, Pedersen (2012) - "Time Series Momentum"
    """

    enabled: bool = False
    target_annual_vol_pct: float = Field(10.0, ge=1.0, le=50.0)
    lookback_days: int = Field(63, ge=10, le=252)
    max_leverage: float = Field(3.0, ge=1.0, le=10.0)   # cap on vol-target scaling
    min_leverage: float = Field(0.1, ge=0.01, le=1.0)    # floor on vol-target scaling


class RiskConfig(BaseModel):
    """Risk manager parameters.

    When enabled, every trade is gated through:
        - ATR trailing stop
        - Position sizer (equity + ADV constraints)
        - Drawdown circuit breaker (NORMAL -> WARNING -> HALTED)
        - Exposure limits (gross, net, single-name)
    """

    enabled: bool = False
    stop_multiplier: float = Field(3.0, ge=1.0, le=10.0)
    stop_max_loss_pct: float = Field(15.0, ge=1.0, le=50.0)
    sizer_max_pct_equity: float = Field(10.0, ge=1.0, le=50.0)
    sizer_max_pct_adv: float = Field(10.0, ge=1.0, le=50.0)
    drawdown_warning_pct: float = Field(-5.0, le=0.0)
    drawdown_halt_pct: float = Field(-15.0, le=0.0)
    exposure_max_gross_pct: float = Field(200.0, ge=50.0, le=500.0)
    exposure_max_net_pct: float = Field(50.0, ge=5.0, le=200.0)
    exposure_max_single_name_pct: float = Field(20.0, ge=1.0, le=100.0)


class RegimeConfig(BaseModel):
    """Regime detection parameters."""

    enabled: bool = False
    lookback_days: int = Field(21, ge=5, le=252)


class OutputConfig(BaseModel):
    """Output and reporting settings."""

    tearsheet_path: str | None = None
    markdown_tearsheet: bool = True           # generate .md alongside .html
    results_dir: str = "results"              # base results directory
    run_log_path: str = "results/run_log.json"  # persistent run log
    verbose: bool = False


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """Complete backtest configuration — single source of truth.

    All interfaces (CLI, YAML, TUI, Streamlit) produce a RunConfig,
    which is passed to run_backtest() for execution.
    """

    initial_capital: float = Field(1_000_000.0, ge=1000.0)
    max_position_pct: float = Field(10.0, ge=1.0, le=100.0)
    signal_threshold: float = Field(0.1, ge=0.0, le=1.0)
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "weekly"

    signal: SignalConfig = Field(default_factory=SignalConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    fill: FillConfig = Field(default_factory=FillConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    vol_target: VolTargetConfig = Field(default_factory=VolTargetConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # ----- Factory methods -----

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> RunConfig:
        """Build RunConfig from argparse Namespace (backward-compatible).

        Maps the flat CLI flags to the nested config model.
        """
        tickers = [t.strip().upper() for t in args.universe.split(",")]

        # Sizing config from CLI flags
        sizing_mode = getattr(args, "sizing_mode", "signal")
        sizing = SizingConfig(
            mode=sizing_mode,
            fixed_dollar_per_position=getattr(args, "fixed_dollar", None),
            fixed_shares_per_position=getattr(args, "fixed_shares", None),
        )

        # Vol target from CLI flags
        vol_target_pct = getattr(args, "vol_target", None)
        vol_target = VolTargetConfig(
            enabled=vol_target_pct is not None,
            target_annual_vol_pct=vol_target_pct or 10.0,
            lookback_days=getattr(args, "vol_lookback", 63),
        )

        return cls(
            initial_capital=args.capital,
            max_position_pct=args.max_position_pct,
            signal_threshold=0.1,
            rebalance_frequency=args.rebalance,
            signal=SignalConfig(type=args.signal),
            data=DataConfig(
                universe=tickers,
                start_date=date.fromisoformat(args.start) if args.start else None,
                end_date=date.fromisoformat(args.end) if args.end else None,
                live=args.live,
                provider=args.provider,
            ),
            fill=FillConfig(model=args.fill),
            execution=ExecutionConfig(
                slippage_bps=getattr(args, "slippage_bps", 3.0),
                commission_rate=getattr(args, "commission_rate", 0.005),
                borrow_gc_rate=getattr(args, "borrow_gc_rate", 0.0025),
            ),
            sizing=sizing,
            vol_target=vol_target,
            risk=RiskConfig(
                enabled=args.risk,
                sizer_max_pct_equity=args.max_position_pct,
                exposure_max_single_name_pct=args.max_position_pct * 2,
            ),
            regime=RegimeConfig(enabled=args.regime),
            output=OutputConfig(
                tearsheet_path=args.tearsheet,
                verbose=getattr(args, "verbose", False),
                results_dir=getattr(args, "results_dir", "results"),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> RunConfig:
        """Load RunConfig from a YAML file."""
        from btconfig.yaml_loader import load_config
        return load_config(path)

    def to_yaml(self, path: str) -> None:
        """Save RunConfig to a YAML file."""
        from btconfig.yaml_loader import dump_config
        dump_config(self, path)
