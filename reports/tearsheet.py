"""Tear sheet — single-page HTML report for a backtest run.

Generates a self-contained HTML file with:
    - Summary statistics table
    - Equity curve (SVG inline chart)
    - Drawdown chart
    - Monthly returns heatmap
    - Rolling Sharpe ratio
    - Regime timeline (if available)
    - Cost breakdown
    - Trade log summary

No external dependencies — uses inline SVG for charts.
Open the HTML file in any browser.

Usage:
    from reports.tearsheet import generate_tearsheet
    engine.run(prices)
    generate_tearsheet(engine, output_path="results/tearsheet.html")
"""

from __future__ import annotations

import html
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_tearsheet(
    engine: Any,
    output_path: str = "tearsheet.html",
    title: str | None = None,
    data_source_info: dict | None = None,
) -> str:
    """Generate an HTML tear sheet from a completed backtest engine.

    Args:
        engine: BacktestEngine instance (must have been run).
        output_path: Where to save the HTML file.
        title: Optional title (defaults to signal name).
        data_source_info: Optional dict with data provenance (provider,
            date range, tickers resolved, limitations).

    Returns:
        Path to the generated HTML file.
    """
    if not engine.snapshots:
        raise ValueError("Engine has no snapshots — did you call engine.run()?")

    summary = engine.summary()
    title = title or f"Backtest: {summary.get('signal_name', 'Unknown Signal')}"

    # Extract data series
    dates = [s.date.isoformat() for s in engine.snapshots]
    equity = [s.equity for s in engine.snapshots]
    returns = [s.daily_return for s in engine.snapshots]
    drawdowns = _compute_drawdowns(equity)
    regimes = [s.regime for s in engine.snapshots]
    circuit_states = [s.circuit_state for s in engine.snapshots]

    # Monthly returns
    monthly = _compute_monthly_returns(engine.snapshots)

    # Rolling Sharpe (63-day)
    rolling_sharpe = _rolling_sharpe(returns, window=63)

    # Build HTML
    html_parts = [
        _html_header(title),
        _data_source_banner(data_source_info),
        _summary_table(summary),
        _benchmark_section(summary),
        _risk_decomposition_section(summary),
        _equity_chart(dates, equity, title="Equity Curve"),
        _drawdown_chart(dates, drawdowns),
        _rolling_sharpe_chart(dates, rolling_sharpe),
        _monthly_heatmap(monthly),
        _regime_timeline(dates, regimes) if any(r is not None for r in regimes) else "",
        _cost_breakdown(summary),
        _trade_summary(engine.trades),
        _html_footer(),
    ]

    html_content = "\n".join(html_parts)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_content, encoding="utf-8")

    logger.info("Tear sheet saved to %s", output)
    return str(output)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _compute_drawdowns(equity: list[float]) -> list[float]:
    """Compute drawdown series from equity curve."""
    peak = equity[0]
    dd = []
    for e in equity:
        peak = max(peak, e)
        dd.append((e - peak) / peak * 100 if peak > 0 else 0)
    return dd


def _compute_monthly_returns(snapshots: list) -> dict[tuple[int, int], float]:
    """Compute monthly returns from daily snapshots.

    Returns: {(year, month): return_pct}
    """
    monthly: dict[tuple[int, int], list[float]] = {}
    for s in snapshots:
        key = (s.date.year, s.date.month)
        if key not in monthly:
            monthly[key] = []
        monthly[key].append(s.daily_return)

    return {
        k: ((1 + np.array(v)).prod() - 1) * 100
        for k, v in monthly.items()
    }


def _rolling_sharpe(returns: list[float], window: int = 63) -> list[float | None]:
    """Compute rolling Sharpe ratio."""
    result: list[float | None] = []
    for i in range(len(returns)):
        if i < window:
            result.append(None)
        else:
            w = returns[i - window:i]
            mean = np.mean(w)
            std = np.std(w)
            if std > 0:
                result.append(float(mean / std * np.sqrt(252)))
            else:
                result.append(0.0)
    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _html_header(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0d1117; color: #c9d1d9; padding: 24px;
        max-width: 1200px; margin: 0 auto;
    }}
    h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 24px; }}
    h2 {{ color: #8b949e; margin: 24px 0 12px; font-size: 16px;
          text-transform: uppercase; letter-spacing: 1px; }}
    .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
             gap: 12px; margin-bottom: 24px; }}
    .stat {{
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; text-align: center;
    }}
    .stat .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; }}
    .stat .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .stat .value.positive {{ color: #3fb950; }}
    .stat .value.negative {{ color: #f85149; }}
    .stat .value.neutral {{ color: #c9d1d9; }}
    .chart-container {{
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; margin-bottom: 24px; overflow-x: auto;
    }}
    svg {{ max-width: 100%; }}
    table {{
        width: 100%; border-collapse: collapse;
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    }}
    th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #21262d;
              font-size: 13px; }}
    th {{ color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .heatmap {{ display: grid; gap: 2px; }}
    .heatmap-cell {{
        width: 60px; height: 30px; display: flex; align-items: center;
        justify-content: center; border-radius: 4px; font-size: 11px; font-weight: 600;
    }}
    .regime-bar {{ height: 20px; display: inline-block; }}
    .regime-low {{ background: #238636; }}
    .regime-normal {{ background: #1f6feb; }}
    .regime-elevated {{ background: #d29922; }}
    .regime-crisis {{ background: #f85149; }}
    footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #21262d;
              color: #484f58; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">Generated by backtest-lab</p>
"""


def _html_footer() -> str:
    return """
<footer>backtest-lab tear sheet</footer>
</body>
</html>"""


def _data_source_banner(info: dict | None) -> str:
    """Generate a data source provenance banner."""
    if not info:
        return ""

    provider = info.get("data_provider", "Unknown")
    freq = info.get("data_frequency", "daily")
    start = info.get("data_start", "")
    end = info.get("data_end", "")
    resolved = info.get("data_tickers_resolved", 0)
    requested = info.get("data_tickers_requested", 0)
    limitations = info.get("data_limitations", "")

    ticker_text = f"{resolved}/{requested} tickers resolved" if requested else ""
    date_text = f"{start} &rarr; {end}" if start and end else ""

    parts = [p for p in [provider, f"{freq} data", date_text, ticker_text] if p]
    banner_line = " &nbsp;|&nbsp; ".join(parts)

    limitation_html = ""
    if limitations:
        limitation_html = (
            f'<div style="color:#8b949e;font-size:11px;margin-top:4px;">'
            f'{html.escape(limitations)}</div>'
        )

    return f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
    padding:12px 16px;margin-bottom:20px;">
<div style="color:#58a6ff;font-size:13px;font-weight:600;">
&#128225; {banner_line}
</div>
{limitation_html}
</div>"""


def _summary_table(s: dict) -> str:
    """Build the summary statistics grid."""

    def _val_class(v):
        if isinstance(v, (int, float)):
            if v > 0:
                return "positive"
            elif v < 0:
                return "negative"
        return "neutral"

    stats = [
        ("Total Return", f"{s.get('total_return_pct', 0):+.2f}%", s.get('total_return_pct', 0)),
        ("Ann. Return", f"{s.get('annualized_return_pct', 0):+.2f}%", s.get('annualized_return_pct', 0)),
        ("Sharpe", f"{s.get('sharpe_ratio', 0):.2f}", s.get('sharpe_ratio', 0)),
        ("Sortino", f"{s.get('sortino_ratio', 0):.2f}", s.get('sortino_ratio', 0)),
        ("Max DD", f"{s.get('max_drawdown_pct', 0):.2f}%", s.get('max_drawdown_pct', 0)),
        ("Calmar", f"{s.get('calmar_ratio', 0):.2f}", s.get('calmar_ratio', 0)),
        ("Ann. Vol", f"{s.get('annualized_vol_pct', 0):.2f}%", 0),
        ("Total Trades", f"{s.get('total_trades', 0):,}", 0),
        ("Total Costs", f"${s.get('total_costs', 0):,.0f}", -s.get('total_costs', 0)),
        ("Stops", f"{s.get('total_stops_triggered', 0)}", 0),
        ("Rejected", f"{s.get('total_trades_rejected', 0)}", 0),
        ("Risk Mgr", s.get('risk_manager', 'OFF'), 0),

        # Trade analytics
        ("Win Rate", f"{s.get('win_rate', 0):.1f}%", s.get('win_rate', 0)),
        ("Slugging Pct", f"{s.get('slugging_pct', 0):.2f}", s.get('slugging_pct', 0) - 1),
        ("Profit Factor", f"{s.get('profit_factor', 0):.2f}", s.get('profit_factor', 0) - 1),
        ("Avg Win", f"${s.get('avg_win', 0):,.0f}", s.get('avg_win', 0)),
        ("Avg Loss", f"${s.get('avg_loss', 0):,.0f}", -s.get('avg_loss', 0)),
        ("Best Trade", f"${s.get('best_trade', 0):,.0f}", s.get('best_trade', 0)),
        ("Worst Trade", f"${s.get('worst_trade', 0):,.0f}", s.get('worst_trade', 0)),

        # Drawdown duration
        ("Max DD Duration", f"{s.get('max_dd_duration_days', 0)} days", -s.get('max_dd_duration_days', 0)),
        ("Max DD Recovery", f"{s.get('max_dd_recovery_days', 0)} days", -s.get('max_dd_recovery_days', 0)),
    ]

    cards = []
    for label, display, val in stats:
        cls = _val_class(val)
        cards.append(
            f'<div class="stat"><div class="label">{label}</div>'
            f'<div class="value {cls}">{display}</div></div>'
        )

    return f'<h2>Summary</h2>\n<div class="grid">{"".join(cards)}</div>'


def _benchmark_section(s: dict) -> str:
    """Generate benchmark comparison section."""
    if "beta" not in s or s.get("beta") == 0.0 and s.get("benchmark_return_pct") == 0.0:
        return ""

    stats = [
        ("Benchmark", s.get("benchmark_ticker", "SPY"), 0),
        ("Benchmark Return", f"{s.get('benchmark_return_pct', 0):+.2f}%", s.get('benchmark_return_pct', 0)),
        ("Relative Return", f"{s.get('relative_return_pct', 0):+.2f}%", s.get('relative_return_pct', 0)),
        ("Beta", f"{s.get('beta', 0):.3f}", 0),
        ("Alpha", f"{s.get('alpha_pct', 0):+.2f}%", s.get('alpha_pct', 0)),
        ("Info Ratio", f"{s.get('information_ratio', 0):.3f}", s.get('information_ratio', 0)),
        ("Tracking Error", f"{s.get('tracking_error_pct', 0):.2f}%", 0),
        ("Up Capture", f"{s.get('up_capture_pct', 0):.1f}%", s.get('up_capture_pct', 0) - 100),
        ("Down Capture", f"{s.get('down_capture_pct', 0):.1f}%", -(s.get('down_capture_pct', 0) - 100)),
        ("Correlation", f"{s.get('benchmark_correlation', 0):.3f}", 0),
    ]

    # Build the same card-style grid as _summary_table
    cards = []
    for label, display, val in stats:
        cls = "positive" if isinstance(val, (int, float)) and val > 0 else ("negative" if isinstance(val, (int, float)) and val < 0 else "neutral")
        cards.append(
            f'<div class="stat"><div class="label">{label}</div>'
            f'<div class="value {cls}">{display}</div></div>'
        )

    return f'<h2>Benchmark Comparison (SPY)</h2>\n<div class="grid">{"".join(cards)}</div>'


def _risk_decomposition_section(s: dict) -> str:
    """Generate risk decomposition section."""
    if "risk_ann_vol_pct" not in s:
        return ""

    stats = [
        ("Ann. Vol", f"{s.get('risk_ann_vol_pct', 0):.2f}%", 0),
        ("Downside Vol", f"{s.get('risk_downside_vol_pct', 0):.2f}%", 0),
        ("21d Vol", f"{s.get('risk_vol_21d_current_pct', 0):.2f}%", 0),
        ("63d Vol", f"{s.get('risk_vol_63d_current_pct', 0):.2f}%", 0),
        ("Avg Gross", f"{s.get('risk_avg_gross_exposure_pct', 0):.1f}%", 0),
        ("Avg Net", f"{s.get('risk_avg_net_exposure_pct', 0):+.1f}%", 0),
        ("Final Gross", f"{s.get('risk_final_gross_pct', 0):.1f}%", 0),
        ("Final Net", f"{s.get('risk_final_net_pct', 0):+.1f}%", 0),
        ("HHI", f"{s.get('risk_hhi', 0):.4f}", 0),
        ("Top 1 Name", f"{s.get('risk_top1_pct', 0):.1f}%", 0),
        ("Top 5 Names", f"{s.get('risk_top5_pct', 0):.1f}%", 0),
        ("Avg Positions", f"{s.get('risk_avg_positions', 0):.1f}", 0),
    ]

    # Beta decomposition (only if benchmark available)
    beta = s.get("risk_portfolio_beta")
    if beta is not None:
        stats.extend([
            ("Portfolio Beta", f"{beta:.3f}", 0),
            ("Beta P&L", f"{s.get('risk_beta_pnl_pct', 0):+.2f}%", s.get('risk_beta_pnl_pct', 0)),
            ("Alpha P&L", f"{s.get('risk_alpha_pnl_pct', 0):+.2f}%", s.get('risk_alpha_pnl_pct', 0)),
            ("Idio Ratio", f"{s.get('risk_idio_ratio_pct', 0):.1f}%", 0),
            ("Idio Vol", f"{s.get('risk_idio_vol_pct', 0):.2f}%" if s.get('risk_idio_vol_pct') is not None else "N/A", 0),
            ("Idio Sharpe", f"{s.get('risk_idio_sharpe', 0):.3f}" if s.get('risk_idio_sharpe') is not None else "N/A", s.get('risk_idio_sharpe', 0) if s.get('risk_idio_sharpe') is not None else 0),
        ])

    cards = []
    for label, display, val in stats:
        cls = "positive" if isinstance(val, (int, float)) and val > 0 else ("negative" if isinstance(val, (int, float)) and val < 0 else "neutral")
        cards.append(
            f'<div class="stat"><div class="label">{label}</div>'
            f'<div class="value {cls}">{display}</div></div>'
        )

    return f'<h2>Risk Decomposition</h2>\n<div class="grid">{"".join(cards)}</div>'


def _equity_chart(dates: list[str], equity: list[float], title: str = "Equity") -> str:
    """Generate inline SVG equity curve."""
    w, h = 1100, 300
    pad_l, pad_r, pad_t, pad_b = 80, 20, 30, 40

    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    min_e, max_e = min(equity), max(equity)
    if max_e == min_e:
        max_e = min_e + 1

    n = len(dates)
    points = []
    for i, e in enumerate(equity):
        x = pad_l + (i / max(n - 1, 1)) * chart_w
        y = pad_t + chart_h - ((e - min_e) / (max_e - min_e)) * chart_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    # Y-axis labels
    y_labels = ""
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = min_e + frac * (max_e - min_e)
        y = pad_t + chart_h - frac * chart_h
        y_labels += (
            f'<text x="{pad_l - 8}" y="{y + 4}" '
            f'fill="#8b949e" font-size="11" text-anchor="end">'
            f'${val:,.0f}</text>\n'
            f'<line x1="{pad_l}" y1="{y}" x2="{w - pad_r}" y2="{y}" '
            f'stroke="#21262d" stroke-width="1"/>\n'
        )

    svg = f"""<div class="chart-container">
<h2>{title}</h2>
<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
{y_labels}
<polyline points="{polyline}" fill="none" stroke="#58a6ff" stroke-width="1.5"/>
</svg>
</div>"""
    return svg


def _drawdown_chart(dates: list[str], drawdowns: list[float]) -> str:
    """Generate inline SVG drawdown chart."""
    w, h = 1100, 200
    pad_l, pad_r, pad_t, pad_b = 80, 20, 30, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    min_dd = min(drawdowns) if drawdowns else -10
    min_dd = min(min_dd, -1)  # at least -1% range

    n = len(dates)
    points = [f"{pad_l},{pad_t}"]  # start at zero line
    for i, dd in enumerate(drawdowns):
        x = pad_l + (i / max(n - 1, 1)) * chart_w
        y = pad_t + (dd / min_dd) * chart_h if min_dd != 0 else pad_t
        points.append(f"{x:.1f},{y:.1f}")
    points.append(f"{pad_l + chart_w},{pad_t}")  # close to zero

    polygon = " ".join(points)

    svg = f"""<div class="chart-container">
<h2>Drawdown</h2>
<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
<line x1="{pad_l}" y1="{pad_t}" x2="{w - pad_r}" y2="{pad_t}"
      stroke="#8b949e" stroke-width="1"/>
<text x="{pad_l - 8}" y="{pad_t + 4}" fill="#8b949e" font-size="11"
      text-anchor="end">0%</text>
<text x="{pad_l - 8}" y="{pad_t + chart_h + 4}" fill="#8b949e" font-size="11"
      text-anchor="end">{min_dd:.1f}%</text>
<polygon points="{polygon}" fill="rgba(248,81,73,0.3)" stroke="#f85149" stroke-width="1"/>
</svg>
</div>"""
    return svg


def _rolling_sharpe_chart(dates: list[str], sharpe: list[float | None]) -> str:
    """Generate inline SVG rolling Sharpe chart."""
    w, h = 1100, 200
    pad_l, pad_r, pad_t, pad_b = 80, 20, 30, 40
    chart_w = w - pad_l - pad_r
    chart_h = h - pad_t - pad_b

    valid = [s for s in sharpe if s is not None]
    if not valid:
        return ""

    min_s = min(min(valid), -1)
    max_s = max(max(valid), 1)
    range_s = max_s - min_s
    if range_s == 0:
        range_s = 1

    n = len(dates)
    points = []
    for i, s in enumerate(sharpe):
        if s is None:
            continue
        x = pad_l + (i / max(n - 1, 1)) * chart_w
        y = pad_t + chart_h - ((s - min_s) / range_s) * chart_h
        points.append(f"{x:.1f},{y:.1f}")

    if not points:
        return ""

    polyline = " ".join(points)

    # Zero line
    zero_y = pad_t + chart_h - ((0 - min_s) / range_s) * chart_h

    svg = f"""<div class="chart-container">
<h2>Rolling Sharpe (63-day)</h2>
<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
<line x1="{pad_l}" y1="{zero_y}" x2="{w - pad_r}" y2="{zero_y}"
      stroke="#484f58" stroke-width="1" stroke-dasharray="4"/>
<text x="{pad_l - 8}" y="{zero_y + 4}" fill="#8b949e" font-size="11"
      text-anchor="end">0</text>
<polyline points="{polyline}" fill="none" stroke="#d29922" stroke-width="1.5"/>
</svg>
</div>"""
    return svg


def _monthly_heatmap(monthly: dict[tuple[int, int], float]) -> str:
    """Generate monthly returns heatmap as HTML table."""
    if not monthly:
        return ""

    years = sorted(set(y for y, m in monthly))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    rows = ""
    for year in years:
        cells = f"<td>{year}</td>"
        annual = 1.0
        for m in range(1, 13):
            val = monthly.get((year, m))
            if val is not None:
                annual *= (1 + val / 100)
                color = _heatmap_color(val)
                cells += f'<td style="background:{color};color:#fff;font-weight:600">{val:+.1f}%</td>'
            else:
                cells += '<td style="color:#484f58">—</td>'
        annual_pct = (annual - 1) * 100
        color = _heatmap_color(annual_pct)
        cells += f'<td style="background:{color};color:#fff;font-weight:700">{annual_pct:+.1f}%</td>'
        rows += f"<tr>{cells}</tr>\n"

    header = "<th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Annual</th>"

    return f"""<h2>Monthly Returns</h2>
<table>
<thead><tr>{header}</tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _heatmap_color(val: float) -> str:
    """Get a color for the heatmap based on return value."""
    if val >= 5:
        return "#238636"
    elif val >= 2:
        return "#2ea043"
    elif val >= 0:
        return "#1a7f37"
    elif val >= -2:
        return "#da3633"
    elif val >= -5:
        return "#f85149"
    else:
        return "#b62324"


def _regime_timeline(dates: list[str], regimes: list[str | None]) -> str:
    """Generate regime timeline as colored bar."""
    if not any(r is not None for r in regimes):
        return ""

    n = len(dates)
    w = 1100
    pad_l, pad_r = 80, 20
    bar_w = w - pad_l - pad_r

    bars = ""
    for i, r in enumerate(regimes):
        if r is None:
            continue
        x = pad_l + (i / max(n - 1, 1)) * bar_w
        cls = f"regime-{r.replace('_vol', '').replace('_', '-')}"
        bars += f'<rect x="{x:.1f}" y="10" width="{max(bar_w / n, 1):.1f}" height="20" class="{cls}"/>\n'

    # Legend
    legend = (
        '<text x="80" y="50" fill="#8b949e" font-size="11">'
        '<tspan fill="#238636">LOW</tspan>  '
        '<tspan fill="#1f6feb">NORMAL</tspan>  '
        '<tspan fill="#d29922">ELEVATED</tspan>  '
        '<tspan fill="#f85149">CRISIS</tspan>'
        '</text>'
    )

    return f"""<div class="chart-container">
<h2>Market Regime</h2>
<svg viewBox="0 0 {w} 60" xmlns="http://www.w3.org/2000/svg">
{bars}
{legend}
</svg>
</div>"""


def _cost_breakdown(s: dict) -> str:
    """Generate cost breakdown table."""
    costs = [
        ("Commission", s.get("total_commission", 0)),
        ("Spread", s.get("total_spread_cost", 0)),
        ("Market Impact", s.get("total_impact_cost", 0)),
        ("Slippage", s.get("total_slippage_cost", 0)),
        ("Borrow", s.get("total_borrow_cost", 0)),
        ("Total", s.get("total_costs", 0)),
    ]

    rows = ""
    for label, val in costs:
        weight = "font-weight:700" if label == "Total" else ""
        rows += f'<tr style="{weight}"><td>{label}</td><td>${val:,.2f}</td></tr>\n'

    return f"""<h2>Execution Costs</h2>
<table style="max-width:400px">
<thead><tr><th>Category</th><th>Amount</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _trade_summary(trades: list) -> str:
    """Generate trade log summary (first 50 trades)."""
    if not trades:
        return "<h2>Trades</h2><p style='color:#8b949e'>No trades executed.</p>"

    rows = ""
    for t in trades[:50]:
        color = "#3fb950" if t.side == "buy" else "#f85149"
        decision = html.escape(t.risk_decision[:30]) if hasattr(t, 'risk_decision') else ""
        rows += (
            f'<tr>'
            f'<td>{t.date}</td>'
            f'<td>{t.ticker}</td>'
            f'<td style="color:{color}">{t.side.upper()}</td>'
            f'<td>{t.shares:,}</td>'
            f'<td>${t.fill_price:.2f}</td>'
            f'<td>${t.total_cost:.2f}</td>'
            f'<td>{t.signal_score:+.2f}</td>'
            f'<td>{decision}</td>'
            f'</tr>\n'
        )

    note = f"<p style='color:#8b949e;margin-top:8px'>Showing first 50 of {len(trades):,} trades</p>" if len(trades) > 50 else ""

    return f"""<h2>Trade Log</h2>
<table>
<thead><tr>
<th>Date</th><th>Ticker</th><th>Side</th><th>Shares</th>
<th>Fill</th><th>Cost</th><th>Signal</th><th>Risk</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
{note}"""
