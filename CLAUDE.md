# algotrader

Backtesting different algorithmic trading strategies on US equity 1-minute IEX bars
(HF Data Library). Read `README.md` (usage, design, results) and `STRATEGY.md`
(exact rules) before changing code. **Read `HANDOFF.md` first**, and `ROADMAP.md` for the research-based priority list of
what to implement next (keep it updated as items are done): it records the work in
progress (runs on the VPS, what is committed and what is not, next steps).

@README.md
@STRATEGY.md

## Working rules

- Code lives in `algotrader/` (object-oriented: `BarLoader`, `CostModel`,
  `PositionSizer`, strategy classes in `strategy.py`, `tree.py`, `ml.py`, `intraday.py`,
  `meanrev.py`, `momentum.py`, overlays in `overlays.py`, the portfolio engine and
  cross-sectional strategies in `portfolio.py`, `factors.py`, `buffered.py` and `trend.py`, `Backtester`, `Metrics`, `stats.py`, `french.py` (Fama-French factor regressions), `history.py` (long daily history), and
  the `experiments.py` runner). Add new strategies as
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
  changed) and commit `results/` in the same commit. Each experiment's fingerprint
  covers only the code it uses (runner without the registry, engine/cost/metric
  modules, its strategy module and that module's package imports, its own definition),
  so adding a strategy does not invalidate the others; `results/summary.md` marks an
  experiment "(stale)" when its fingerprint no longer matches. Do not commit stale
  rows. Do not edit `algotrader/*.py` in a checkout that experiments are running from
  (develop in a scratch copy and copy the files back after committing the results).
- Tune only on the training period (the runner enforces this); report test results of
  every experiment, not only the best ones.
- Never introduce look-ahead: signals use bar close, fills use next open. Models that
  learn (e.g. `DecisionTreeStrategy.fit`) train only on bars before the test period.

## Environment

- Dependencies are managed with uv: `pyproject.toml` + `uv.lock` + `.python-version`.
  `uv sync` builds `.venv/`; add packages with `uv add` / `uv add --dev` and commit the
  updated `uv.lock`. Do not `pip install` into `.venv/` by hand. `data/` is git-ignored.
- Heavy experiment runs (1-minute bars, many tickers) go on the Hermes VPS (12 GB RAM;
  ask the `Hermes` session for connection details, never copy its key or `~/.hermes`).
  Work only in `~/algotrader-work` there: rsync the working tree (without `.venv`, `data`,
  `results`), `uv sync`, run detached with `nohup`, rsync `results/` back, commit here.
  This machine has little free memory: use `--workers 1` locally (Claude Code also
  stops idle background jobs here when memory is low). To stop a run on the VPS, kill
  it by PID and also kill its spawned worker processes (they survive the parent);
  `pkill -f algotrader.experiments` inside an ssh command also matches the ssh shell.
- Data download needs an HF Data Library API key stored OUTSIDE the repo at
  `~/.config/hfdatalibrary/token`. Never print, commit or write the key into any file.
- Long daily history (`history.py`) uses a Tiingo token in `~/.config/tiingo/token` and a
  Kaggle token in `~/.kaggle/access_token` (both outside the repo, chmod 600, never print
  them). Tiingo's free quota is per account (500 symbols a month): download only once
  (on the VPS) and rsync `data/history/` to wherever it is needed.
- Run scripts with data under `data/scratch/tickers/<TICKER>_clean.parquet`.

## Legacy code

The old agent-written scripts and results were deleted (still in git history up to `7fc720e`).
Their published numbers never reproduced; do not cite them.
