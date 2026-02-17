"""Matplotlib chart generation for markdown tearsheets.

Produces PNG files with a dark theme matching the HTML tearsheet.
All functions take data + output_path and return the saved file path.

Uses Agg backend for headless rendering (no display required).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import date

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark theme matching the HTML tearsheet
# ---------------------------------------------------------------------------

_DARK_BG = "#0d1117"
_PANEL_BG = "#161b22"
_GRID_COLOR = "#21262d"
_TEXT_COLOR = "#c9d1d9"
_MUTED_COLOR = "#8b949e"
_BLUE = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_ORANGE = "#d29922"
_DPI = 150


def _apply_dark_style(fig, ax):
    """Apply dark theme to figure and axes."""
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_MUTED_COLOR, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.xaxis.label.set_color(_MUTED_COLOR)
    ax.yaxis.label.set_color(_MUTED_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.5, alpha=0.7)


def _parse_dates(date_strings: list[str]) -> list[date]:
    """Parse ISO date strings to date objects."""
    return [date.fromisoformat(d) for d in date_strings]


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plot_equity_curve(
    dates: list[str], equity: list[float], output_path: str, title: str = "Equity Curve"
) -> str:
    """Plot equity curve and save as PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_style(fig, ax)

    dt = _parse_dates(dates)
    ax.plot(dt, equity, color=_BLUE, linewidth=1.2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved equity curve: %s", output_path)
    return output_path


def plot_drawdown(
    dates: list[str], drawdowns: list[float], output_path: str
) -> str:
    """Plot drawdown chart and save as PNG."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _apply_dark_style(fig, ax)

    dt = _parse_dates(dates)
    ax.fill_between(dt, drawdowns, 0, color=_RED, alpha=0.3)
    ax.plot(dt, drawdowns, color=_RED, linewidth=1.0)
    ax.axhline(0, color=_MUTED_COLOR, linewidth=0.8)
    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved drawdown chart: %s", output_path)
    return output_path


def plot_rolling_sharpe(
    dates: list[str], sharpe: list[float | None], output_path: str, window: int = 63
) -> str:
    """Plot rolling Sharpe ratio and save as PNG."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _apply_dark_style(fig, ax)

    # Filter out None values
    dt_vals = []
    sharpe_vals = []
    for d, s in zip(dates, sharpe):
        if s is not None:
            dt_vals.append(date.fromisoformat(d))
            sharpe_vals.append(s)

    if not sharpe_vals:
        plt.close(fig)
        return output_path

    ax.plot(dt_vals, sharpe_vals, color=_ORANGE, linewidth=1.2)
    ax.axhline(0, color=_MUTED_COLOR, linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling Sharpe ({window}-day)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved rolling Sharpe: %s", output_path)
    return output_path


def plot_monthly_heatmap(
    monthly_returns: dict[tuple[int, int], float], output_path: str
) -> str:
    """Plot monthly returns heatmap and save as PNG."""
    if not monthly_returns:
        return output_path

    years = sorted(set(y for y, m in monthly_returns))
    list(range(1, 13))

    data = np.full((len(years), 12), np.nan)
    for (y, m), val in monthly_returns.items():
        row = years.index(y)
        data[row, m - 1] = val

    fig, ax = plt.subplots(figsize=(10, max(2, len(years) * 0.6 + 1)))
    _apply_dark_style(fig, ax)

    # Custom colormap: red -> dark -> green
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "returns", [_RED, _PANEL_BG, _GREEN], N=256
    )

    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 5.0)
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    # Labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=9, color=_MUTED_COLOR)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=9, color=_MUTED_COLOR)

    # Annotate cells
    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "#ffffff" if abs(val) > vmax * 0.3 else _TEXT_COLOR
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_title("Monthly Returns", fontsize=14, fontweight="bold", color=_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved monthly heatmap: %s", output_path)
    return output_path


def plot_regime_timeline(
    dates: list[str], regimes: list[str | None], output_path: str
) -> str:
    """Plot regime timeline as colored bar and save as PNG."""
    if not any(r is not None for r in regimes):
        return output_path

    regime_colors = {
        "low_vol": _GREEN,
        "low": _GREEN,
        "normal": _BLUE,
        "normal_vol": _BLUE,
        "elevated_vol": _ORANGE,
        "elevated": _ORANGE,
        "crisis": _RED,
        "crisis_vol": _RED,
    }

    fig, ax = plt.subplots(figsize=(10, 1.5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    dt = _parse_dates(dates)
    n = len(dt)

    for i, (d, r) in enumerate(zip(dt, regimes)):
        if r is None:
            continue
        color = regime_colors.get(r.lower(), _MUTED_COLOR)
        width = (dt[-1] - dt[0]).days / n if n > 1 else 1
        ax.barh(0, width, left=mdates.date2num(d), color=color, height=0.8, edgecolor="none")

    ax.set_yticks([])
    ax.set_title("Market Regime", fontsize=14, fontweight="bold", color=_TEXT_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(colors=_MUTED_COLOR, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(_GRID_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved regime timeline: %s", output_path)
    return output_path
