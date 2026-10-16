"""Reproducible experiments: tune on the training period, evaluate out of sample.

Every experiment runs over the whole ticker universe with the same split:

    train  TRAIN_START .. TEST_START   parameters are chosen here only
    test   TEST_START .. end of data   reported out of sample

Rule-based strategies are tuned by backtesting each grid point on the training
period. Strategies that learn (`fit="ticker"`: one model per ticker, or
`fit="pooled"`: one model on all tickers' bars) are fitted before VALID_START and
the grid is scored on VALID_START .. TEST_START; the chosen setting is then refitted
on the whole training period. The test period is also run without costs ("gross")
to show how much of any edge costs take. A grid point is only eligible if its median
ticker made at least MIN_TRADES trades in the scoring window, so a setting that
barely trades cannot win on a lucky handful of trades. The selection score is the Sharpe of the
equal-weight portfolio of all tickers (each ticker's equity normalised to 1).

Usage:
    python -m algotrader.experiments --list
    python -m algotrader.experiments                  # run all
    python -m algotrader.experiments orb_5min rsi2_daily --workers 3

Writes to results/: <name>.csv (per ticker, train and test, chosen parameters),
<name>_grid.csv (tuning scores), <name>_equity.csv (daily test equity per ticker),
manifest.json (code and data fingerprints per experiment) and summary.md.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .costs import CostModel
from .data import BarLoader
from .engine import Backtester
from .intraday import (IntradayMomentumStrategy, NoiseAreaStrategy,
                       OpeningRangeBreakoutStrategy, OvernightStrategy)
from .meanrev import RsiReversionStrategy
from .ml import GradientBoostingStrategy
from .metrics import TRADING_DAYS, Metrics
from .sizing import PositionSizer
from .strategy import SmaCrossStrategy
from .tree import DecisionTreeStrategy
from .universe import UNIVERSE

TRAIN_START = "2022-03-01"
VALID_START = "2023-07-01"
TEST_START = "2024-01-01"
CAPITAL = 10_000.0
MIN_TRADES = 5
IBKR = CostModel()                                           # $0.005/share, $1 minimum
FULL = PositionSizer(risk_per_trade=1.0, max_position_frac=1.0)   # fully invested
PACKAGE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Experiment:
    """One strategy, bar size, cost model and parameter grid."""
    name: str
    description: str
    strategy: type
    timeframe: str
    params: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)
    costs: CostModel = IBKR
    sizer: PositionSizer = FULL
    fit: str = ""                 # "", "ticker" (fit per ticker) or "pooled" (one model)
    capital: float = CAPITAL      # starting equity per ticker

    def combos(self) -> list:
        """Parameter dicts to try: `params` updated with each point of `grid`."""
        keys = list(self.grid)
        out = []
        for values in itertools.product(*(self.grid[k] for k in keys)):
            p = {**self.params, **dict(zip(keys, values))}
            try:
                self.strategy(**p)                   # skip invalid points (e.g. fast >= slow)
            except ValueError:
                continue
            out.append(p)
        return out or [dict(self.params)]


EXPERIMENTS = [
    Experiment("sma_daily", "SMA crossover on daily bars (trend following)",
               SmaCrossStrategy, "1D", {"atr_period": 14},
               {"fast": [10, 20, 50], "slow": [50, 100, 200], "atr_mult": [3.0, 50.0]}),
    Experiment("tree_1min", "Decision tree (arXiv:2405.13959), paper settings, 1-min bars",
               DecisionTreeStrategy, "1min", {"max_depth": 4, "atr_mult": 50.0}, fit="ticker"),
    Experiment("tree_30min", "Decision tree, 30-min bars, depth tuned",
               DecisionTreeStrategy, "30min", {"atr_mult": 50.0},
               {"max_depth": [2, 4, 6]}, fit="ticker"),
    Experiment("overnight_1min", "Hold from the last minute to the next open",
               OvernightStrategy, "1min", {}, {"trend_days": [None, 20, 50, 200]}),
    Experiment("intraday_momentum_5min",
               "Gao et al. (2018): last half hour after an up first half hour",
               IntradayMomentumStrategy, "5min", {},
               {"min_ret": [0.0, 0.0025, 0.005], "require_r12": [False, True]}),
    Experiment("noise_area_1min", "Zarattini et al. (2024) noise-area breakout, long only",
               NoiseAreaStrategy, "1min", {"lookback": 14},
               {"band_mult": [0.5, 1.0, 1.5, 2.0], "decision_minutes": [30, 60]}),
    Experiment("orb_5min", "Zarattini & Aziz (2023) opening-range breakout, long only",
               OpeningRangeBreakoutStrategy, "5min", {},
               {"or_minutes": [5, 15, 30], "min_rel_volume": [0.0, 1.5, 2.5]}),
    Experiment("rsi2_daily", "Connors RSI(2) mean reversion on daily bars",
               RsiReversionStrategy, "1D", {"rsi_period": 2},
               {"entry": [5.0, 10.0, 20.0], "exit_sma": [5, 10], "trend_sma": [None, 200]}),
    Experiment("gbm_30min", "Gradient boosting, cost-aware label, pooled over tickers",
               GradientBoostingStrategy, "30min", {},
               {"horizon": [4, 13], "threshold": [0.002, 0.005], "entry_quantile": [0.8, 0.9]},
               fit="pooled"),
    Experiment("gbm_daily", "Gradient boosting on daily bars, pooled over tickers",
               GradientBoostingStrategy, "1D", {},
               {"horizon": [1, 5], "threshold": [0.0, 0.01], "entry_quantile": [0.7, 0.85]},
               fit="pooled"),
    Experiment("overnight_1min_100k", "Overnight hold with $100,000 per ticker (cost share)",
               OvernightStrategy, "1min", {}, {"trend_days": [None, 20, 50, 200]},
               capital=100_000.0),
    Experiment("orb_5min_100k", "Opening-range breakout with $100,000 per ticker (cost share)",
               OpeningRangeBreakoutStrategy, "5min", {},
               {"or_minutes": [5, 15, 30], "min_rel_volume": [0.0, 1.5, 2.5]},
               capital=100_000.0),
]
BY_NAME = {e.name: e for e in EXPERIMENTS}


# --------------------------------------------------------------------- running

def _backtest(exp, params, bars, phase, model=None):
    """Run one phase on one ticker's bars: 'grid' (tuning), 'train', 'test' or
    'gross' (test without costs). `model` is a pre-fitted pooled strategy."""
    train = bars[bars.index < TEST_START]
    strat = model if model is not None else exp.strategy(**params)
    costs = CostModel(0.0, 0.0) if phase == "gross" else exp.costs
    engine = Backtester(strat, costs, exp.sizer, exp.capital)
    if exp.fit == "ticker":
        strat.fit(train[train.index < VALID_START] if phase == "grid" else train)
    if phase == "grid" and exp.fit:              # fitted models: score on validation only
        return engine.run(train, start=VALID_START), train[train.index >= VALID_START]
    if phase in ("grid", "train"):
        return engine.run(train), train
    return engine.run(bars, start=TEST_START), bars[bars.index >= TEST_START]


def _train_frames(exp, data_dir, tickers) -> list:
    """Every ticker's training bars (before TEST_START), for pooled fitting."""
    loader = BarLoader(data_dir)
    frames = []
    for t in tickers:
        bars = loader.load(t, exp.timeframe)
        frames.append(bars[bars.index < TEST_START])
    return frames


def _fit_pooled(exp, params, frames, end):
    """Fit one strategy on all frames, using only bars before `end`."""
    return exp.strategy(**params).fit_many([f[f.index < end] for f in frames])


def _daily_curve(values: pd.Series, start_value: float) -> pd.Series:
    """Daily last values normalised by `start_value`, with 1.0 on the day before the first."""
    daily = values.resample("1D").last().dropna() / start_value
    first = pd.Series([1.0], index=[daily.index[0] - pd.Timedelta("1D")])
    return pd.concat([first, daily])


def _run_ticker(args):
    """Worker: load one ticker once and run every (phase, params) job on it."""
    exp_name, ticker, data_dir, jobs = args
    exp = BY_NAME[exp_name]
    bars = BarLoader(data_dir).load(ticker, exp.timeframe)
    out = []
    for phase, k, params, model in jobs:
        result, period_bars = _backtest(exp, params, bars, phase, model)
        summary = Metrics(result, period_bars).summary()
        daily = _daily_curve(result.equity, exp.capital)
        hold = _daily_curve(period_bars["close"], period_bars["open"].iloc[0])
        out.append((phase, k, ticker, summary, daily, hold))
    return out


def portfolio_stats(curves: dict) -> dict:
    """Equal-weight portfolio of normalised daily curves (rebalanced daily)."""
    df = pd.DataFrame(curves).sort_index().ffill()
    r = df.pct_change().mean(axis=1).dropna()
    eq = (1 + r).cumprod()
    years = (df.index[-1] - df.index[0]).days / 365.25 if len(df) > 1 else 0
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else float("nan")
    total = float(eq.iloc[-1] - 1) if len(eq) else float("nan")
    return {"return": total,
            "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan"),
            "sharpe": sharpe,
            "max_drawdown": float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")}


def _map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn, jobs))


def run_experiment(exp: Experiment, data_dir, tickers, workers=3) -> dict:
    """Tune on train, then run train and test with the chosen parameters."""
    combos = exp.combos()
    grid_rows, best, note = [], 0, ""
    pooled = exp.fit == "pooled"
    frames = _train_frames(exp, data_dir, tickers) if pooled else None
    if len(combos) > 1:
        models = [_fit_pooled(exp, p, frames, VALID_START) if pooled else None
                  for p in combos]
        jobs = [(exp.name, t, str(data_dir),
                 [("grid", k, p, models[k]) for k, p in enumerate(combos)]) for t in tickers]
        curves = {k: {} for k in range(len(combos))}
        sharpes = {k: [] for k in range(len(combos))}
        trades = {k: [] for k in range(len(combos))}
        for chunk in _map(_run_ticker, jobs, workers):
            for _, k, ticker, summary, daily, _ in chunk:
                curves[k][ticker] = daily
                sharpes[k].append(summary["sharpe"])
                trades[k].append(summary["trades"])
        for k, p in enumerate(combos):
            stats = portfolio_stats(curves[k])
            med_trades = float(np.median(trades[k]))
            grid_rows.append({"params": json.dumps(p, sort_keys=True), **{
                f"portfolio_{s}": v for s, v in stats.items()},
                "median_ticker_sharpe": float(np.nanmedian(sharpes[k]))
                if not np.isnan(sharpes[k]).all() else np.nan,
                "median_trades": med_trades, "eligible": med_trades >= MIN_TRADES})
        scores = np.array([r["portfolio_sharpe"] for r in grid_rows], dtype=float)
        eligible = np.array([r["eligible"] for r in grid_rows])
        if eligible.any():
            scores[~eligible] = np.nan
        else:
            note = f"no grid point reached {MIN_TRADES} median trades; best of all shown"
        best = int(np.nanargmax(np.nan_to_num(scores, nan=-np.inf)))
    params = combos[best]
    model = _fit_pooled(exp, params, frames, TEST_START) if pooled else None
    del frames
    phases = ("train", "test", "gross")
    jobs = [(exp.name, t, str(data_dir), [(ph, 0, params, model) for ph in phases])
            for t in tickers]
    rows, curves, holds = [], {ph: {} for ph in phases}, {ph: {} for ph in phases}
    for chunk in _map(_run_ticker, jobs, workers):
        for phase, _, ticker, summary, daily, hold in chunk:
            rows.append({"ticker": ticker, "period": phase, **summary})
            curves[phase][ticker], holds[phase][ticker] = daily, hold
    return {"params": params, "note": note, "rows": pd.DataFrame(rows),
            "grid": pd.DataFrame(grid_rows),
            "portfolio": {ph: portfolio_stats(curves[ph]) for ph in curves},
            "hold": {ph: portfolio_stats(holds[ph]) for ph in holds},
            "test_equity": pd.DataFrame(curves["test"]).sort_index()}


# --------------------------------------------------------------- fingerprints

def code_fingerprint() -> str:
    """sha256 over the package source files (name + content), first 16 hex digits."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def data_fingerprint(data_dir, tickers) -> dict:
    out = {}
    for t in tickers:
        path = BarLoader(data_dir).path(t)
        out[t] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                  "bytes": path.stat().st_size}
    return out


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=PACKAGE_DIR, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", str(PACKAGE_DIR)))}


# --------------------------------------------------------------------- outputs

def write_results(exp, res, out_dir, manifest_entry):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = res["rows"].copy()
    rows.insert(2, "params", json.dumps(res["params"], sort_keys=True))
    rows.to_csv(out / f"{exp.name}.csv", index=False, float_format="%.6g")
    if not res["grid"].empty:
        res["grid"].to_csv(out / f"{exp.name}_grid.csv", index=False, float_format="%.6g")
    res["test_equity"].to_csv(out / f"{exp.name}_equity.csv", float_format="%.6g")
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[exp.name] = manifest_entry
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _pct(x):
    return "" if pd.isna(x) else f"{x:+.1%}"


def _num(x):
    return "" if pd.isna(x) else f"{x:.2f}"


def write_summary(out_dir):
    """Regenerate summary.md from the manifest (one row per experiment)."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    current = code_fingerprint()
    lines = [
        "# Experiment results", "",
        "Generated by `python -m algotrader.experiments`; do not edit by hand.",
        f"Split: train {TRAIN_START} to {TEST_START} (parameters chosen here), "
        f"test {TEST_START} to end of data (out of sample). Equal-weight portfolio of all "
        "tickers, fully invested when in a position, IBKR commissions. "
        "B&H = buy and hold of the same tickers over the same period. A grid point needs a "
        f"median of {MIN_TRADES} trades per ticker in its scoring window to be chosen.", "",
        "| Experiment | Bars | Capital/ticker | Chosen parameters | Test return | Test Sharpe | Test max DD "
        "| Before costs: return, Sharpe | B&H return | B&H Sharpe | Train Sharpe "
        "| Trades/yr per ticker | Time in market | Code |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in [e.name for e in EXPERIMENTS if e.name in manifest]:
        m = manifest[name]
        te, tr, bh = m["portfolio"]["test"], m["portfolio"]["train"], m["hold"]["test"]
        gr = m["portfolio"]["gross"]
        stale = "" if m["code_sha256"] == current else " (stale)"
        params = ", ".join(f"{k}={v}" for k, v in sorted(m["params"].items())) or "defaults"
        if m.get("selection_note"):
            params += f" ({m['selection_note']})"
        lines.append(
            f"| `{name}` | {m['timeframe']} | ${m.get('capital', CAPITAL):,.0f} | {params} | {_pct(te['return'])} | {_num(te['sharpe'])} "
            f"| {_pct(te['max_drawdown'])} | {_pct(gr['return'])}, {_num(gr['sharpe'])} "
            f"| {_pct(bh['return'])} | {_num(bh['sharpe'])} "
            f"| {_num(tr['sharpe'])} | {m['test_trades_per_year']:.0f} | {m['test_exposure']:.0%} "
            f"| `{m['code_sha256'][:8]}`{stale} |")
    lines += ["", "Per-ticker numbers: `<experiment>.csv`; tuning grid: `<experiment>_grid.csv`; "
              "daily test equity: `<experiment>_equity.csv`; fingerprints: `manifest.json`.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="experiments to run (default: all)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--data-dir", default="data/scratch/tickers")
    p.add_argument("--tickers", nargs="*", help="default: algotrader.universe.UNIVERSE")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    if args.list:
        for e in EXPERIMENTS:
            print(f"{e.name:24s} {e.timeframe:6s} {len(e.combos()):3d} combos  {e.description}")
        return
    unknown = [n for n in args.names if n not in BY_NAME]
    if unknown:
        p.error(f"unknown experiments: {unknown}")
    tickers = args.tickers or UNIVERSE
    data = data_fingerprint(args.data_dir, tickers)
    for exp in [BY_NAME[n] for n in args.names] if args.names else EXPERIMENTS:
        res = run_experiment(exp, args.data_dir, tickers, args.workers)
        test = res["rows"][res["rows"]["period"] == "test"]
        years = (pd.Timestamp(res["test_equity"].index[-1]) - pd.Timestamp(TEST_START)).days / 365.25
        entry = {"description": exp.description, "strategy": exp.strategy.__name__,
                 "timeframe": exp.timeframe, "params": res["params"],
                 "grid": {k: [str(v) for v in vs] for k, vs in exp.grid.items()},
                 "costs": vars(exp.costs), "sizer": vars(exp.sizer), "fit": exp.fit,
                 "capital": exp.capital, "selection_note": res["note"],
                 "split": {"train_start": TRAIN_START, "valid_start": VALID_START,
                           "test_start": TEST_START},
                 "portfolio": res["portfolio"], "hold": res["hold"],
                 "test_trades_per_year": float(test["trades"].mean() / years) if years > 0 else 0.0,
                 "test_exposure": float(test["exposure"].mean()),
                 "code_sha256": code_fingerprint(), "git": git_state(),
                 "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "data": data}
        write_results(exp, res, args.out, entry)
        write_summary(args.out)
        te, bh = res["portfolio"]["test"], res["hold"]["test"]
        print(f"{exp.name}: params {res['params']} | test {te['return']:+.1%} "
              f"Sharpe {te['sharpe']:.2f} (B&H {bh['return']:+.1%}, {bh['sharpe']:.2f})", flush=True)


if __name__ == "__main__":
    main()
