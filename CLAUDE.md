# algotrader

Backtesting different algorithmic trading strategies on US equity 1-minute IEX bars
(HF Data Library). Read `README.md` (usage, design, results) and `STRATEGY.md`
(exact rules) before changing code.

@README.md
@STRATEGY.md

## Working rules

- Code lives in `algotrader/` (object-oriented: `BarLoader`, `CostModel`,
  `PositionSizer`, strategy classes in `strategy.py`, `tree.py`, `ml.py`, `intraday.py`,
  `meanrev.py`, `Backtester`, `Metrics`, and the `experiments.py` runner). Add new strategies as
  new classes satisfying the `Strategy` protocol (`indicators(bars)` returning
  `want_long` and `atr`, plus an `atr_mult` attribute; `average_true_range` is
  available); do not copy-paste engine code.
- Every behaviour change needs a unit test in `tests/`. Run `uv run pytest`
  before committing. Keep tests on small hand-built series (`tests/helpers.py`).
- Document changes: update README (design/tests/results tables) and STRATEGY.md in the
  same commit. Do not state performance numbers that have not been produced by this code.
- **Results are committed with the code that produced them.** When adding or changing a
  strategy, the engine, costs or the runner, register it in `EXPERIMENTS`
  (`algotrader/experiments.py`), rerun the affected experiments
  (`uv run python -m algotrader.experiments <names>`; all of them if shared code
  changed) and commit `results/` in the same commit. `results/summary.md` marks an
  experiment "(stale)" when its code fingerprint no longer matches the current code;
  do not commit stale rows. Do not edit `algotrader/*.py` while experiments run (the
  fingerprint is taken when each finishes).
- Tune only on the training period (the runner enforces this); report test results of
  every experiment, not only the best ones.
- Never introduce look-ahead: signals use bar close, fills use next open. Models that
  learn (e.g. `DecisionTreeStrategy.fit`) train only on bars before the test period.

## Environment

- Dependencies are managed with uv: `pyproject.toml` + `uv.lock` + `.python-version`.
  `uv sync` builds `.venv/`; add packages with `uv add` / `uv add --dev` and commit the
  updated `uv.lock`. Do not `pip install` into `.venv/` by hand. `data/` is git-ignored.
- Data download needs an HF Data Library API key stored OUTSIDE the repo at
  `~/.config/hfdatalibrary/token`. Never print, commit or write the key into any file.
- Run scripts with data under `data/scratch/tickers/<TICKER>_clean.parquet`.

## Legacy code

The old agent-written scripts and results were deleted (still in git history up to `7fc720e`).
Their published numbers never reproduced; do not cite them.
