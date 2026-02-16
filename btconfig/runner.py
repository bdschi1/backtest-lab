"""Shared backtest runner — single entry point for all interfaces.

All interfaces (CLI, YAML, TUI, Streamlit) build a RunConfig and
call run_backtest(config). This function handles:
    1. Resolving default dates
    2. Fetching or generating data
    3. Instantiating signals, fill models, cost models
    4. Building risk manager, vol-target scaler, and regime detector
    5. Running the engine
    6. Creating results directory with HTML + MD tearsheets
    7. Appending to persistent run log
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from btconfig.run_config import RunConfig

logger = logging.getLogger(__name__)


def run_backtest(config: RunConfig) -> tuple[Any, dict]:
    """Execute a backtest from a RunConfig.

    Args:
        config: Fully specified RunConfig.

    Returns:
        (engine, summary_dict) — engine has .snapshots, .trades, etc.
    """
    from engine.backtest import BacktestConfig, BacktestEngine

    # 1. Resolve dates
    end_date = config.data.end_date or date.today()
    start_date = config.data.start_date or (end_date - timedelta(days=3 * 365))

    # 2. Fetch or generate data
    data_source_info: dict[str, Any] = {
        "data_provider": "Synthetic",
        "data_frequency": "daily",
        "data_start": str(start_date),
        "data_end": str(end_date),
        "data_tickers_requested": len(config.data.universe),
        "data_tickers_resolved": 0,
        "data_tickers_failed": 0,
        "data_failed_tickers": [],
        "data_limitations": "",
    }

    if config.data.live:
        prices, data_source_info = _fetch_live_tracked(
            config.data.universe, start_date, end_date, config.data.provider
        )
    else:
        prices = _generate_synthetic(config.data.universe, start_date, end_date)
        data_source_info["data_provider"] = "Synthetic"
        data_source_info["data_tickers_resolved"] = len(config.data.universe)
        data_source_info["data_limitations"] = (
            "Simulated price data (GBM). Not based on real market prices. "
            "For demonstration and testing only."
        )

    if prices.height == 0:
        raise ValueError("No data available for the specified universe and dates.")

    # 3. Build signal
    signal = _build_signal(config)

    # 4. Build fill model
    fill = _build_fill_model(config)

    # 5. Build cost models (slippage, commission, borrow)
    slippage, commission, borrow = _build_cost_models(config)

    # 6. Build BacktestConfig
    bt_config = BacktestConfig(
        initial_capital=config.initial_capital,
        max_position_pct=config.max_position_pct,
        signal_threshold=config.signal_threshold,
        rebalance_frequency=config.rebalance_frequency,
        sizing_mode=config.sizing.mode,
        fixed_dollar_per_position=config.sizing.fixed_dollar_per_position,
        fixed_shares_per_position=config.sizing.fixed_shares_per_position,
    )

    # 7. Build risk manager
    risk_manager = _build_risk_manager(config) if config.risk.enabled else None

    # 8. Build vol-target scaler
    vol_target_scaler = _build_vol_target(config)

    # 9. Build regime detector
    regime_detector, regime_adapter = _build_regime(config)

    # 10. Create engine
    engine = BacktestEngine(
        signal=signal,
        fill_model=fill,
        slippage_model=slippage,
        commission_model=commission,
        borrow_model=borrow,
        config=bt_config,
        risk_manager=risk_manager,
        regime_detector=regime_detector,
        regime_adapter=regime_adapter,
        vol_target_scaler=vol_target_scaler,
    )

    # 11. Run
    engine.run(prices)
    summary = engine.summary() or {}

    # 11b. Benchmark comparison (SPY)
    benchmark_returns = None
    try:
        from risk.benchmark import compute_benchmark_metrics, fetch_benchmark_returns
        strategy_returns = [s.daily_return for s in engine.snapshots]
        # Fetch SPY for the same date range
        benchmark_returns = fetch_benchmark_returns(start_date, end_date)
        if benchmark_returns is not None:
            # Strategy returns include the lookback warmup period with 0 returns.
            # Align: benchmark returns are from day 1, strategy from day 0.
            # Both are daily returns, trim to match.
            bench_metrics = compute_benchmark_metrics(strategy_returns, benchmark_returns)
            summary.update(bench_metrics)
        else:
            logger.info("Benchmark data unavailable — skipping comparison metrics")
    except Exception as e:
        logger.warning("Failed to compute benchmark metrics: %s", e)

    # 11c. Risk decomposition (uses benchmark returns if available)
    try:
        from risk.risk_decomposition import RiskDecomposition
        rd = RiskDecomposition(engine, benchmark_returns=benchmark_returns)
        risk_summary = rd.compute_summary()
        summary.update(risk_summary)
    except Exception as e:
        logger.warning("Failed to compute risk decomposition: %s", e)

    # 12. Create results directory and generate outputs
    run_dir = _create_run_dir(config)

    # 13. Generate HTML tearsheet
    html_path = config.output.tearsheet_path or str(Path(run_dir) / "tearsheet.html")
    from reports.tearsheet import generate_tearsheet
    generate_tearsheet(
        engine, output_path=html_path, data_source_info=data_source_info,
    )
    summary["tearsheet_html"] = html_path

    # 14. Generate markdown tearsheet
    if config.output.markdown_tearsheet:
        try:
            from reports.markdown_tearsheet import generate_markdown_tearsheet
            md_path = str(Path(run_dir) / "tearsheet.md")
            generate_markdown_tearsheet(
                engine, output_dir=run_dir, output_path=md_path,
                data_source_info=data_source_info,
            )
            summary["tearsheet_md"] = md_path
        except Exception as e:
            logger.warning("Failed to generate markdown tearsheet: %s", e)

    # 15. Save config snapshot
    try:
        config.to_yaml(str(Path(run_dir) / "config.yaml"))
    except Exception as e:
        logger.warning("Failed to save config YAML: %s", e)

    # 16. Attach data source metadata (before run log so log can read them)
    summary.update(data_source_info)
    summary["run_dir"] = run_dir

    # 17. Append to run log
    try:
        _append_run_log(config, summary, run_dir)
    except Exception as e:
        logger.warning("Failed to append to run log: %s", e)

    return engine, summary


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build_signal(config: RunConfig):
    """Instantiate the signal from btconfig."""
    from signals.momentum import (
        CompositeSignal,
        MeanReversionSignal,
        MomentumSignal,
    )

    sc = config.signal

    if sc.type == "momentum":
        return MomentumSignal(
            formation_days=sc.momentum_formation_days,
            skip_days=sc.momentum_skip_days,
            zscore_clip=sc.momentum_zscore_clip,
        )
    elif sc.type == "mean_reversion":
        return MeanReversionSignal(
            lookback_days=sc.mean_reversion_lookback_days,
            zscore_clip=sc.mean_reversion_zscore_clip,
        )
    else:
        # Composite — build sub-signals and combine
        mom = MomentumSignal(
            formation_days=sc.momentum_formation_days,
            skip_days=sc.momentum_skip_days,
            zscore_clip=sc.momentum_zscore_clip,
        )
        mr = MeanReversionSignal(
            lookback_days=sc.mean_reversion_lookback_days,
            zscore_clip=sc.mean_reversion_zscore_clip,
        )
        weights = sc.composite_weights
        return CompositeSignal([
            (mom, weights.get("momentum", 0.6)),
            (mr, weights.get("mean_reversion", 0.4)),
        ])


def _build_fill_model(config: RunConfig):
    """Instantiate the fill model from btconfig."""
    from execution.fill_model import (
        MarketImpactFill,
        MidPriceFill,
        SpreadAwareFill,
    )

    fc = config.fill

    if fc.model == "spread":
        return SpreadAwareFill(default_spread_bps=fc.spread_default_bps)
    elif fc.model == "impact":
        return MarketImpactFill(eta=fc.impact_eta)
    else:
        return MidPriceFill()


def _build_cost_models(config: RunConfig):
    """Instantiate slippage, commission, and borrow models.

    When fill model is 'mid' (prototyping tier), all costs are zero.
    Otherwise, costs are based on config.execution parameters.
    """
    if config.fill.model == "mid":
        return None, None, None

    from execution.slippage import FixedSlippage
    from execution.commission import PerShareCommission
    from execution.borrow import TieredBorrow

    ec = config.execution
    slippage = FixedSlippage(bps=ec.slippage_bps)
    commission = PerShareCommission(rate=ec.commission_rate)
    borrow = TieredBorrow(gc_rate=ec.borrow_gc_rate)

    return slippage, commission, borrow


def _build_risk_manager(config: RunConfig):
    """Build the risk manager from btconfig."""
    from risk.risk_manager import RiskManager
    from risk.stop_loss import ATRTrailingStop
    from risk.position_sizer import PositionSizer
    from risk.drawdown_control import DrawdownController
    from risk.exposure_limits import ExposureLimits

    rc = config.risk

    return RiskManager(
        stop_loss=ATRTrailingStop(
            multiplier=rc.stop_multiplier,
            max_loss_pct=rc.stop_max_loss_pct,
        ),
        sizer=PositionSizer(
            max_pct_equity=rc.sizer_max_pct_equity,
            max_pct_adv=rc.sizer_max_pct_adv,
        ),
        drawdown=DrawdownController(
            warning_dd_pct=rc.drawdown_warning_pct,
            halt_dd_pct=rc.drawdown_halt_pct,
        ),
        exposure=ExposureLimits(
            max_gross_pct=rc.exposure_max_gross_pct,
            max_net_pct=rc.exposure_max_net_pct,
            max_single_name_pct=rc.exposure_max_single_name_pct,
        ),
    )


def _build_vol_target(config: RunConfig):
    """Build vol-target scaler if enabled."""
    if not config.vol_target.enabled:
        return None

    from risk.vol_target import VolTargetScaler

    vt = config.vol_target
    return VolTargetScaler(
        target_vol_pct=vt.target_annual_vol_pct,
        lookback_days=vt.lookback_days,
        max_leverage=vt.max_leverage,
        min_leverage=vt.min_leverage,
    )


def _build_regime(config: RunConfig):
    """Build regime detector and adapter from btconfig."""
    if not config.regime.enabled:
        return None, None

    from regime.detector import VolatilityRegimeDetector
    from regime.adapter import RegimeAdapter

    detector = VolatilityRegimeDetector(lookback_days=config.regime.lookback_days)
    adapter = RegimeAdapter()

    return detector, adapter


# ---------------------------------------------------------------------------
# Results directory and run log
# ---------------------------------------------------------------------------

def _create_run_dir(config: RunConfig) -> str:
    """Create timestamped results directory."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    signal_type = config.signal.type
    run_dir = Path(config.output.results_dir) / f"{ts}_{signal_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _append_run_log(config: RunConfig, summary: dict, run_dir: str):
    """Append run metadata to persistent JSON log."""
    log_path = Path(config.output.run_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    entries: list[dict] = []
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text())
        except (json.JSONDecodeError, ValueError):
            entries = []

    entry = {
        "run_id": Path(run_dir).name,
        "timestamp": datetime.now().isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": _serialize_summary(summary),
        "data_source": {
            "provider": summary.get("data_provider", "Unknown"),
            "frequency": summary.get("data_frequency", "daily"),
            "date_range": f"{summary.get('data_start', '')} → {summary.get('data_end', '')}",
            "tickers_resolved": summary.get("data_tickers_resolved", 0),
            "tickers_failed": summary.get("data_tickers_failed", 0),
            "failed_tickers": summary.get("data_failed_tickers", []),
            "limitations": summary.get("data_limitations", ""),
        },
        "output_dir": run_dir,
        "files": {
            "html": str(Path(run_dir) / "tearsheet.html"),
            "markdown": str(Path(run_dir) / "tearsheet.md"),
            "config": str(Path(run_dir) / "config.yaml"),
        },
    }

    entries.append(entry)
    log_path.write_text(json.dumps(entries, indent=2, default=str))
    logger.info("Run log updated: %s", log_path)


def _serialize_summary(summary: dict) -> dict:
    """Serialize summary dict for JSON storage."""
    result = {}
    for k, v in summary.items():
        if isinstance(v, (date, datetime)):
            result[k] = v.isoformat()
        elif isinstance(v, (int, float, str, bool, type(None))):
            result[k] = v
        elif isinstance(v, np.floating):
            result[k] = float(v)
        elif isinstance(v, np.integer):
            result[k] = int(v)
        else:
            result[k] = str(v)
    return result


# ---------------------------------------------------------------------------
# Data helpers (moved from cli.py)
# ---------------------------------------------------------------------------

_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "yahoo": "Yahoo Finance",
    "bloomberg": "Bloomberg Terminal",
    "ib": "Interactive Brokers",
    "refinitiv": "Refinitiv / LSEG",
    "polygon": "Polygon.io",
    "databento": "Databento",
}

_PROVIDER_LIMITATIONS: dict[str, str] = {
    "yahoo": (
        "Daily adjusted close only. No intraday, no real-time, no bid/ask. "
        "Data may be delayed 15-20 min. Adjusted close accounts for splits "
        "and dividends."
    ),
    "bloomberg": (
        "Full intraday + EOD with bid/ask spreads, corporate actions, and "
        "reference data. Requires Bloomberg Terminal subscription and blpapi."
    ),
    "ib": (
        "Real-time streaming + historical intraday with bid/ask. "
        "Requires Interactive Brokers account and TWS/IB Gateway running."
    ),
    "refinitiv": "Tick data, reference data, economic indicators. Requires LSEG subscription.",
    "polygon": "REST API + WebSocket streaming, trades & quotes, 15+ years history.",
    "databento": "Tick-by-tick, normalized across exchanges, futures/options.",
}

_PROVIDER_FREQUENCIES: dict[str, str] = {
    "yahoo": "daily",
    "bloomberg": "intraday + daily",
    "ib": "intraday + daily",
    "refinitiv": "tick + intraday + daily",
    "polygon": "tick + intraday + daily",
    "databento": "tick + intraday + daily",
}


def _fetch_live_tracked(
    tickers: list[str], start: date, end: date, provider_name: str
) -> tuple[Any, dict]:
    """Fetch live data and return (prices, data_source_info)."""
    info: dict[str, Any] = {
        "data_provider": _PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name),
        "data_frequency": _PROVIDER_FREQUENCIES.get(provider_name, "daily"),
        "data_start": str(start),
        "data_end": str(end),
        "data_tickers_requested": len(tickers),
        "data_tickers_resolved": 0,
        "data_tickers_failed": 0,
        "data_failed_tickers": [],
        "data_limitations": _PROVIDER_LIMITATIONS.get(provider_name, ""),
    }

    prices = _fetch_live(tickers, start, end, provider_name)

    # Track which tickers actually resolved
    if prices.height > 0 and "ticker" in prices.columns:
        fetched = prices["ticker"].unique().to_list()
        resolved = [t for t in tickers if t in fetched]
        failed = [t for t in tickers if t not in fetched]
        info["data_tickers_resolved"] = len(resolved)
        info["data_tickers_failed"] = len(failed)
        info["data_failed_tickers"] = failed
    else:
        info["data_tickers_resolved"] = 0
        info["data_tickers_failed"] = len(tickers)
        info["data_failed_tickers"] = tickers
        info["data_provider"] = "Synthetic (fallback)"
        info["data_limitations"] = (
            f"{_PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name)} "
            "returned no data. Fell back to synthetic data."
        )

    return prices, info


def _fetch_live(tickers: list[str], start: date, end: date, provider_name: str):
    """Fetch live data from the specified provider.

    Implements a fallback chain:
        1. Try the requested provider (yahoo/bloomberg/ib).
        2. If the provider returns data but some tickers are missing,
           log a warning and continue with available tickers.
        3. If the provider fails entirely or returns no data,
           fall back to synthetic data so the run still completes.
    """
    if provider_name == "bloomberg":
        from data.bloomberg_provider import BloombergProvider
        provider = BloombergProvider()
    elif provider_name == "ib":
        from data.ib_provider import IBProvider
        provider = IBProvider()
    else:
        from data.yahoo_provider import YahooProvider
        provider = YahooProvider()

    logger.info("Fetching data from %s...", provider.name)

    try:
        prices = provider.fetch_daily_prices(tickers, start, end)
    except Exception as e:
        logger.warning(
            "%s fetch failed (%s). Falling back to synthetic data.",
            provider.name, e,
        )
        return _generate_synthetic(tickers, start, end)

    if prices.height == 0:
        logger.warning(
            "%s returned no data. Falling back to synthetic data.",
            provider.name,
        )
        return _generate_synthetic(tickers, start, end)

    # Check which tickers actually came back
    if "ticker" in prices.columns:
        fetched_tickers = prices["ticker"].unique().to_list()
        missing = [t for t in tickers if t not in fetched_tickers]
        if missing:
            logger.warning(
                "%s: no data for %d ticker(s): %s. "
                "Backfilling missing tickers with synthetic data.",
                provider.name, len(missing), ", ".join(missing),
            )
            synthetic_backfill = _generate_synthetic(missing, start, end)
            prices = pl.concat([prices, synthetic_backfill], how="diagonal")

    return prices


def _generate_synthetic(tickers: list[str], start: date, end: date):
    """Generate synthetic price data for prototyping."""
    logger.info("Generating synthetic data for %d tickers", len(tickers))
    np.random.seed(42)
    rows = []

    stock_params = {}
    for ticker in tickers:
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
                + vol * np.sqrt(max(days_elapsed, 1)) * np.random.normal()
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
