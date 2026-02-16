"""Bloomberg data provider for backtest-lab.

Re-exports from bds-data-providers shared package.
"""

from bds_data_providers.bloomberg import BloombergProvider, is_available

__all__ = ["BloombergProvider", "is_available"]
