"""Interactive Brokers data provider for backtest-lab.

Re-exports from bds-data-providers shared package.
"""

from bds_data_providers.ib import IBProvider, is_available

__all__ = ["IBProvider", "is_available"]
