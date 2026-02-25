# Contributing

## Development Setup

```bash
git clone https://github.com/bdschi1/backtest-lab.git
cd backtest-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

- Lint with `ruff check .`
- Format with `ruff format .`
- Type hints encouraged

## Testing

```bash
pytest tests/ -v
```

## Pull Requests

1. Create a feature branch from `main`
2. Make focused, single-purpose commits
3. Ensure all tests pass before submitting
4. Open a PR with a clear description of changes
