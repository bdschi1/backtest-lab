"""YAML configuration loader and dumper.

Loads a YAML file into a RunConfig and saves a RunConfig back to YAML.
PyYAML is an optional dependency: pip install backtest-lab[yaml]
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: str) -> "RunConfig":  # noqa: F821
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated RunConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ImportError: If PyYAML is not installed.
        ValidationError: If the config values are invalid.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML configs. "
            "Install it with: pip install backtest-lab[yaml]"
        )

    from btconfig.run_config import RunConfig

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return RunConfig(**raw)


def dump_config(config: "RunConfig", path: str) -> None:  # noqa: F821
    """Save a RunConfig to a YAML file.

    Args:
        config: RunConfig instance to save.
        path: Output file path.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML configs. "
            "Install it with: pip install backtest-lab[yaml]"
        )

    from datetime import date

    data = config.model_dump(exclude_none=True)

    # Convert date objects to ISO strings for clean YAML
    def _convert_dates(obj):
        if isinstance(obj, dict):
            return {k: _convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_dates(v) for v in obj]
        elif isinstance(obj, date):
            return obj.isoformat()
        return obj

    data = _convert_dates(data)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
