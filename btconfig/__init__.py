"""Shared configuration and runner for backtest-lab.

All interfaces (CLI, YAML, TUI, Streamlit) produce a RunConfig,
which is executed by run_backtest().
"""

from btconfig.run_config import (
    DataConfig,
    ExecutionConfig,
    FillConfig,
    OutputConfig,
    RegimeConfig,
    RiskConfig,
    RunConfig,
    SignalConfig,
    SizingConfig,
    VolTargetConfig,
)
from btconfig.runner import run_backtest

__all__ = [
    "DataConfig",
    "ExecutionConfig",
    "FillConfig",
    "OutputConfig",
    "RegimeConfig",
    "RiskConfig",
    "RunConfig",
    "SignalConfig",
    "SizingConfig",
    "VolTargetConfig",
    "run_backtest",
]
