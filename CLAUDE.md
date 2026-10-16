# algotrader

Backtesting different algorithmic trading strategies on US equity 1-minute IEX bars
(HF Data Library). Read `README.md` (usage, design, results) and `STRATEGY.md`
(exact rules) before changing code.

@README.md
@STRATEGY.md

## Working rules

- Code lives in `algotrader/` (object-oriented: `BarLoader`, `CostModel`,
  `PositionSizer`, `SmaCrossStrategy`, `Backtester`, `Metrics`). Add new strategies as
  new classes with an `indicators(bars)` method returning `want_long` and an `atr_mult`
  attribute; do not copy-paste engine code.
- Every behaviour change needs a unit test in `tests/`. Run `.venv/bin/python -m pytest`
  before committing. Keep tests on small hand-built series (`tests/helpers.py`).
- Document changes: update README (design/tests/results tables) and STRATEGY.md in the
  same commit. Do not state performance numbers that have not been produced by this code.
- Never introduce look-ahead: signals use bar close, fills use next open.

## Environment

- Python venv in `.venv/` (`uv venv`, `uv pip install -e ".[dev]"`). `data/` is git-ignored.
- Data download needs an HF Data Library API key stored OUTSIDE the repo at
  `~/.config/hfdatalibrary/token`. Never print, commit or write the key into any file.
- Run scripts with data under `data/scratch/tickers/<TICKER>_clean.parquet`.

## Legacy code

The old agent-written scripts and results were deleted (still in git history up to `7fc720e`).
Their published numbers never reproduced; do not cite them.
