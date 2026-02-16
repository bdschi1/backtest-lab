"""Provider factory — discover, register, and switch data providers.

Same pattern as ls-portfolio-lab and multi-agent-investment-committee.
Auto-detects installed providers; defaults to Yahoo Finance.

Registered providers:
    - Yahoo Finance — always available (free, daily EOD)
    - Bloomberg Terminal — requires blpapi (intraday + EOD, bid/ask)
    - Interactive Brokers — requires ib_insync (real-time, intraday, bid/ask)
    - Refinitiv / LSEG — planned (tick data, reference data)
    - Polygon.io — planned (REST + WebSocket, trades & quotes)
    - Databento — planned (tick-by-tick, normalized)

Usage:
    from data.provider_factory import get_provider, available_providers

    provider = get_provider()                  # Yahoo (default)
    provider = get_provider("Bloomberg")       # Bloomberg Terminal
    provider = get_provider("Interactive Brokers")  # IB TWS/Gateway
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from data.provider import DataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

# (display_name, module_path, class_name, availability_func_path | None)
_PROVIDER_REGISTRY: list[tuple[str, str, str, str | None]] = [
    ("Yahoo Finance", "data.yahoo_provider", "YahooProvider", None),
    (
        "Bloomberg",
        "data.bloomberg_provider",
        "BloombergProvider",
        "data.bloomberg_provider.is_available",
    ),
    (
        "Interactive Brokers",
        "data.ib_provider",
        "IBProvider",
        "data.ib_provider.is_available",
    ),
    # Planned providers — stubs for registry awareness.
    # Implementations not yet available; availability check will fail gracefully.
    (
        "Refinitiv",
        "data.refinitiv_provider",
        "RefinitivProvider",
        "data.refinitiv_provider.is_available",
    ),
    (
        "Polygon.io",
        "data.polygon_provider",
        "PolygonProvider",
        "data.polygon_provider.is_available",
    ),
    (
        "Databento",
        "data.databento_provider",
        "DatabentoProvider",
        "data.databento_provider.is_available",
    ),
]

_provider_cache: dict[str, DataProvider] = {}


def available_providers() -> list[str]:
    """Return names of providers whose dependencies are installed.

    Yahoo Finance is always available.
    """
    available: list[str] = []

    for display_name, _module_path, _class_name, avail_func_path in _PROVIDER_REGISTRY:
        if avail_func_path is None:
            available.append(display_name)
            continue

        try:
            mod_path, func_name = avail_func_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            is_avail = getattr(mod, func_name)
            if is_avail():
                available.append(display_name)
        except Exception:
            pass

    return available


def get_provider(name: str = "Yahoo Finance", **kwargs: Any) -> DataProvider:
    """Get a data provider instance by name.

    Args:
        name: "Yahoo Finance", "Bloomberg", or "Interactive Brokers"
        **kwargs: Passed to provider constructor (e.g., host, port)

    Returns:
        DataProvider instance

    Raises:
        ValueError: unknown provider name
        ImportError: required package not installed
    """
    if name in _provider_cache and not kwargs:
        return _provider_cache[name]

    for display_name, module_path, class_name, _ in _PROVIDER_REGISTRY:
        if display_name == name:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls(**kwargs)
            _provider_cache[name] = instance
            logger.info("Data provider initialized: %s", name)
            return instance

    available = available_providers()
    raise ValueError(f"Unknown provider '{name}'. Available: {available}")


def get_provider_safe(name: str = "Yahoo Finance", **kwargs: Any) -> DataProvider:
    """Get a provider, falling back to Yahoo Finance on any error."""
    try:
        return get_provider(name, **kwargs)
    except Exception as exc:
        logger.warning(
            "Failed to initialize %s (%s), falling back to Yahoo Finance",
            name, exc,
        )
        return get_provider("Yahoo Finance")
