"""Reproducible experiments: walk-forward tuning and out-of-sample evaluation.

Every experiment runs over the whole ticker universe.

Walk-forward (the main result): for each fold start F in FOLD_STARTS, parameters are
chosen using only data before F and then tested on F .. next fold start (the last
fold runs to the end of the data). The folds' test periods are concatenated into one
out-of-sample record.

Single split (kept for continuity with earlier results): chosen with data before
TEST_START, tested on TEST_START .. end of data. Its choice is the same as the fold
starting at TEST_START.

How parameters are chosen at F:
  * rule-based strategies: every grid point is backtested from the start of the data;
    the score is its result before F. Because their indicators are causal, one
    full-history run per grid point serves every fold and test window (a window's
    numbers are that run's equity inside the window).
  * strategies that learn (`fit="ticker"`: one model per ticker; `fit="pooled"`: one
    model on all tickers' bars): fitted on data before F - VALID_MONTHS, scored on
    F - VALID_MONTHS .. F; the chosen setting is refitted on all data before F.
  * score = Sharpe of the equal-weight portfolio of all tickers. A grid point is
    eligible only if its median ticker made at least MIN_TRADES trades in the
    scoring window.

Test windows are run with the experiment's costs ("base": IBKR commission plus a
half-spread of SPREAD_BPS per side), without costs ("gross") and with a wider
half-spread ("stress", STRESS_SPREAD_BPS).

Usage:
    python -m algotrader.experiments --list
    python -m algotrader.experiments                  # run all
    python -m algotrader.experiments orb_5min rsi2_daily --workers 2

Writes to results/: <name>.csv (single split per ticker: train, test, gross, stress),
<name>_folds.csv (walk-forward per fold and ticker), <name>_grid.csv (tuning scores
per fold), <name>_equity.csv (single-split daily test equity per ticker),
<name>_wf_returns.csv (walk-forward daily portfolio returns), manifest.json (code and
data fingerprints per experiment) and summary.md.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .costs import CostModel
from .data import BarLoader
from .engine import Backtester, BacktestResult
from .intraday import (IntradayMomentumStrategy, NoiseAreaStrategy,
                       OpeningRangeBreakoutStrategy, OvernightStrategy)
from .meanrev import RsiReversionStrategy
from .metrics import Metrics
from .ml import GradientBoostingStrategy
from .sizing import PositionSizer
from .stats import (TRADING_DAYS, alpha_beta, expected_max_sharpe, portfolio_returns,
                    returns_stats, sharpe_interval)
from .strategy import SmaCrossStrategy
from .tree import DecisionTreeStrategy
from .universe import UNIVERSE

TEST_START = "2024-01-01"
FOLD_STARTS = ["2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
               "2026-01-01"]
VALID_MONTHS = 6
CAPITAL = 10_000.0
MIN_TRADES = 5
SPREAD_BPS = 2.0
STRESS_SPREAD_BPS = 5.0
BASE_COSTS = CostModel(spread_bps=SPREAD_BPS)                # IBKR + 2 bp half-spread
FULL = PositionSizer(risk_per_trade=1.0, max_position_frac=1.0)   # fully invested
VARIANTS = ("base", "gross", "stress")
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
    costs: CostModel = BASE_COSTS
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

    def costs_for(self, variant: str) -> CostModel:
        if variant == "gross":
            return CostModel(0.0, 0.0)
        if variant == "stress":
            return replace(self.costs, spread_bps=STRESS_SPREAD_BPS)
        return self.costs


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


# ------------------------------------------------------------------- windows

def valid_start(fold_start) -> pd.Timestamp:
    """Start of the validation window for fitted models in the fold starting at F."""
    return pd.Timestamp(fold_start) - pd.DateOffset(months=VALID_MONTHS)


def fold_windows() -> list:
    """[(start, end)] test windows of the walk-forward; the last end is None (data end)."""
    starts = [pd.Timestamp(s) for s in FOLD_STARTS]
    return [(s, starts[i + 1] if i + 1 < len(starts) else None) for i, s in enumerate(starts)]


def selection_starts() -> list:
    """Every date at which parameters are chosen (fold starts and the single split)."""
    return sorted({pd.Timestamp(s) for s in FOLD_STARTS} | {pd.Timestamp(TEST_START)})


def _before(bars, end):
    return bars if end is None else bars[bars.index < end]


def slice_result(result: BacktestResult, start=None, end=None) -> BacktestResult | None:
    """The part of a backtest inside [start, end): equity, positions and the trades
    entered there, with the equity just before `start` as the starting capital."""
    idx = result.equity.index
    mask = np.ones(len(idx), dtype=bool)
    if start is not None:
        mask &= idx >= start
    if end is not None:
        mask &= idx < end
    if not mask.any():
        return None
    before = result.equity[idx < start] if start is not None else result.equity.iloc[:0]
    initial = float(before.iloc[-1]) if len(before) else result.initial_capital
    trades = [t for t in result.trades
              if (start is None or t.entry_time >= start) and (end is None or t.entry_time < end)]
    pos = result.position[mask] if result.position is not None else None
    return BacktestResult(initial, result.equity[mask], trades, pos)


def _daily_curve(values: pd.Series, start_value: float) -> pd.Series:
    """Daily last values normalised by `start_value`, with 1.0 on the day before the first."""
    daily = values.resample("1D").last().dropna() / start_value
    first = pd.Series([1.0], index=[daily.index[0] - pd.Timedelta("1D")])
    return pd.concat([first, daily])


def _window_record(result, bars, start, end):
    """(summary, daily strategy curve, daily buy-and-hold curve) for one window."""
    part = slice_result(result, start, end)
    if part is None:
        return None
    period = bars[(bars.index >= start) if start is not None else slice(None)]
    period = _before(period, end)
    summary = Metrics(part, period).summary()
    return (summary, _daily_curve(part.equity, part.initial_capital),
            _daily_curve(period["close"], period["open"].iloc[0]))


# ------------------------------------------------------------------- workers

def _engine(exp, strat, variant):
    return Backtester(strat, exp.costs_for(variant), exp.sizer, exp.capital)


def _select_ticker(args):
    """Pass 1 worker: score every grid point at every selection date on one ticker."""
    exp_name, ticker, data_dir, starts, combos, models = args
    exp = BY_NAME[exp_name]
    bars = BarLoader(data_dir).load(ticker, exp.timeframe)
    out = []
    if not exp.fit:
        for k, p in enumerate(combos):
            full = _engine(exp, exp.strategy(**p), "base").run(bars)
            for F in starts:
                rec = _window_record(full, bars, None, F)
                if rec:
                    out.append((F, k, ticker, rec[0], rec[1]))
        return out
    for F in starts:
        v = valid_start(F)
        scored = _before(bars, F)
        if not (scored.index >= v).any() or not (scored.index < v).any():
            continue
        for k, p in enumerate(combos):
            if exp.fit == "pooled":
                strat = models[(F, k)]
            else:
                strat = exp.strategy(**p)
                try:
                    strat.fit(bars[bars.index < v])
                except ValueError:
                    continue
            rec = _window_record(_engine(exp, strat, "base").run(scored, start=v), bars, v, F)
            if rec:
                out.append((F, k, ticker, rec[0], rec[1]))
    return out


def _final_ticker(args):
    """Pass 2 worker: run the chosen settings on one ticker's test windows.

    windows: [(name, start, end, params, fit_end)]; fit_end is where a fitted model's
    training stops. Each window is run with every cost variant ("train" only base).
    """
    exp_name, ticker, data_dir, windows, models = args
    exp = BY_NAME[exp_name]
    bars = BarLoader(data_dir).load(ticker, exp.timeframe)
    cache, out = {}, []
    for name, start, end, params, fit_end in windows:
        variants = ("base",) if name == "train" else VARIANTS
        if not exp.fit:
            for var in variants:
                key = (json.dumps(params, sort_keys=True), var)
                if key not in cache:
                    cache[key] = _engine(exp, exp.strategy(**params), var).run(bars)
                rec = _window_record(cache[key], bars, start, end)
                if rec:
                    out.append((name, var, ticker) + rec)
            continue
        if exp.fit == "pooled":
            strat = models[fit_end]
        else:
            strat = exp.strategy(**params)
            try:
                strat.fit(bars[bars.index < fit_end])
            except ValueError:
                continue
        upto = _before(bars, end)
        if start is not None and not (upto.index >= start).any():
            continue
        for var in variants:
            run = _engine(exp, strat, var).run(upto, start=start)
            rec = _window_record(run, bars, start, end)
            if rec:
                out.append((name, var, ticker) + rec)
    return out


def _map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(fn, jobs))


# ------------------------------------------------------------------- driver

def _train_frames(exp, data_dir, tickers, end) -> list:
    """Every ticker's bars before `end`, for pooled fitting."""
    loader = BarLoader(data_dir)
    return [_before(loader.load(t, exp.timeframe), end) for t in tickers]


def _fit_pooled(exp, params, frames, end):
    """Fit one strategy on all frames, using only bars before `end`."""
    return exp.strategy(**params).fit_many([f[f.index < end] for f in frames])


def _choose(grid_rows):
    """Index of the best eligible grid point (by portfolio Sharpe) and a note."""
    scores = np.array([r["portfolio_sharpe"] for r in grid_rows], dtype=float)
    eligible = np.array([r["eligible"] for r in grid_rows], dtype=bool)
    note = ""
    if eligible.any():
        scores[~eligible] = np.nan
    else:
        note = f"no grid point reached {MIN_TRADES} median trades; best of all used"
    return int(np.nanargmax(np.nan_to_num(scores, nan=-np.inf))), note


def run_experiment(exp: Experiment, data_dir, tickers, workers=2) -> dict:
    """Choose parameters at every selection date, then run the test windows."""
    combos = exp.combos()
    starts = selection_starts()
    pooled = exp.fit == "pooled"
    frames = _train_frames(exp, data_dir, tickers, max(starts)) if pooled else None

    # pass 1: selection
    chosen, notes, grid_rows = {F: 0 for F in starts}, {F: "" for F in starts}, []
    if len(combos) > 1:
        models = ({(F, k): _fit_pooled(exp, p, frames, valid_start(F))
                   for F in starts for k, p in enumerate(combos)} if pooled else None)
        jobs = [(exp.name, t, str(data_dir), starts, combos, models) for t in tickers]
        curves = {(F, k): {} for F in starts for k in range(len(combos))}
        trades = {key: [] for key in curves}
        for chunk in _map(_select_ticker, jobs, workers):
            for F, k, ticker, summary, daily in chunk:
                curves[(F, k)][ticker] = daily
                trades[(F, k)].append(summary["trades"])
        for F in starts:
            rows = []
            for k, p in enumerate(combos):
                stats = (returns_stats(portfolio_returns(curves[(F, k)]))
                         if curves[(F, k)] else returns_stats(pd.Series(dtype=float)))
                med = float(np.median(trades[(F, k)])) if trades[(F, k)] else 0.0
                rows.append({"fold": F.date().isoformat(), "params": json.dumps(p, sort_keys=True),
                             **{f"portfolio_{s}": v for s, v in stats.items()},
                             "median_trades": med, "eligible": med >= MIN_TRADES})
            chosen[F], notes[F] = _choose(rows)
            grid_rows += rows

    # pass 2: test windows with the chosen settings
    single = pd.Timestamp(TEST_START)
    windows = [(f"fold:{s.date()}", s, e, combos[chosen[s]], s) for s, e in fold_windows()]
    windows += [("test", single, None, combos[chosen[single]], single),
                ("train", None, single, combos[chosen[single]], single)]
    models = ({F: _fit_pooled(exp, combos[chosen[F]], frames, F) for F in starts}
              if pooled else None)
    del frames
    jobs = [(exp.name, t, str(data_dir), windows, models) for t in tickers]
    recs = [r for chunk in _map(_final_ticker, jobs, workers) for r in chunk]
    return _assemble(exp, combos, chosen, notes, grid_rows, recs)


def _assemble(exp, combos, chosen, notes, grid_rows, recs) -> dict:
    """Portfolio statistics for the single split and the walk-forward."""
    by = {}
    for name, var, ticker, summary, daily, hold in recs:
        by.setdefault((name, var), {})[ticker] = (summary, daily, hold)

    def port(name, var):
        got = by.get((name, var), {})
        return portfolio_returns({t: v[1] for t, v in got.items()}) if got else pd.Series(dtype=float)

    def hold(name):
        got = by.get((name, "base"), {})
        return portfolio_returns({t: v[2] for t, v in got.items()}) if got else pd.Series(dtype=float)

    rows = []
    for (name, var), got in by.items():
        period = {"base": name, "gross": "gross", "stress": "stress"}[var] if name == "test" else None
        if name == "train":
            period = "train"
        for ticker, (summary, _, _) in got.items():
            rows.append({"ticker": ticker, "period": period or f"{name}/{var}", **summary})
    rows = pd.DataFrame(rows)
    single_rows = rows[rows["period"].isin(["train", "test", "gross", "stress"])]
    fold_rows = rows[~rows["period"].isin(["train", "test", "gross", "stress"])]

    folds = [f"fold:{s.date()}" for s, _ in fold_windows()]
    wf = {var: pd.concat([port(f, var) for f in folds]).sort_index() for var in VARIANTS}
    wf_hold = pd.concat([hold(f) for f in folds]).sort_index()
    fold_sharpes = {f.split(":")[1]: returns_stats(port(f, "base"))["sharpe"] for f in folds}
    test_curves = by.get(("test", "base"), {})
    return {
        "params": combos[chosen[pd.Timestamp(TEST_START)]],
        "note": notes[pd.Timestamp(TEST_START)],
        "fold_params": {s.date().isoformat(): combos[chosen[s]] for s, _ in fold_windows()},
        "fold_notes": {s.date().isoformat(): notes[s] for s, _ in fold_windows() if notes[s]},
        "rows": single_rows, "fold_rows": fold_rows, "grid": pd.DataFrame(grid_rows),
        "portfolio": {"train": returns_stats(port("train", "base")),
                      "test": returns_stats(port("test", "base")),
                      "gross": returns_stats(port("test", "gross")),
                      "stress": returns_stats(port("test", "stress"))},
        "hold": {"train": returns_stats(hold("train")), "test": returns_stats(hold("test"))},
        "walk_forward": {
            **{var: returns_stats(wf[var]) for var in VARIANTS},
            "hold": returns_stats(wf_hold),
            "sharpe_90": sharpe_interval(wf["base"]),
            "vs_hold": alpha_beta(wf["base"], wf_hold),
            "fold_sharpe": fold_sharpes,
            "days": int(len(wf["base"])),
        },
        "wf_returns": pd.DataFrame({**{var: wf[var] for var in VARIANTS}, "hold": wf_hold}),
        "test_equity": pd.DataFrame({t: v[1] for t, v in test_curves.items()}).sort_index(),
    }


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
    res["fold_rows"].to_csv(out / f"{exp.name}_folds.csv", index=False, float_format="%.6g")
    if not res["grid"].empty:
        res["grid"].to_csv(out / f"{exp.name}_grid.csv", index=False, float_format="%.6g")
    res["test_equity"].to_csv(out / f"{exp.name}_equity.csv", float_format="%.6g")
    res["wf_returns"].to_csv(out / f"{exp.name}_wf_returns.csv", float_format="%.6g")
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[exp.name] = manifest_entry
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _pct(x):
    return "" if x is None or pd.isna(x) else f"{x:+.1%}"


def _num(x):
    return "" if x is None or pd.isna(x) else f"{x:.2f}"


def _fold_choices(m) -> str:
    """Grid parameters chosen across folds, most common first, e.g. 'a=1 (4/6)'."""
    keys = [k for k in m.get("grid", {})]
    if not keys:
        return "fixed"
    counts = Counter(json.dumps({k: m["fold_params"][f].get(k) for k in keys}, sort_keys=True)
                     for f in m["fold_params"])
    n = len(m["fold_params"])
    parts = []
    for combo, c in counts.most_common():
        txt = ", ".join(f"{k}={v}" for k, v in json.loads(combo).items())
        parts.append(f"{txt} ({c}/{n})")
    return "; ".join(parts)


def write_summary(out_dir):
    """Regenerate summary.md from the manifest."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    current = code_fingerprint()
    names = [e.name for e in EXPERIMENTS if e.name in manifest]
    trials = sum(len(BY_NAME[n].combos()) for n in names)
    any_wf = next((manifest[n]["walk_forward"] for n in names if "walk_forward" in manifest[n]), None)
    years = any_wf["days"] / TRADING_DAYS if any_wf else 0
    hold = any_wf["hold"] if any_wf else {}
    lines = [
        "# Experiment results", "",
        "Generated by `python -m algotrader.experiments`; do not edit by hand. "
        "Equal-weight portfolio of all tickers, fully invested when in a position. "
        f"Costs: IBKR commission + {SPREAD_BPS:g} bp half-spread per side (base); "
        f"none (gross); IBKR + {STRESS_SPREAD_BPS:g} bp (stress). A grid point needs a median "
        f"of {MIN_TRADES} trades per ticker in its scoring window to be chosen.", "",
        "## Walk-forward (main result)", "",
        f"Folds start {', '.join(FOLD_STARTS)}; each is tuned on data before its start and "
        "tested until the next; the test periods are concatenated "
        f"({years:.1f} years). Buy and hold of the same tickers: {_pct(hold.get('return'))}, "
        f"Sharpe {_num(hold.get('sharpe'))}. Alpha/beta: daily returns regressed on buy and "
        f"hold. {trials} configurations were tried in total; the best of that many "
        f"zero-edge strategies would show a Sharpe of about "
        f"{expected_max_sharpe(trials, years):.2f} by luck alone over this length.", "",
        "| Experiment | Bars | Capital/ticker | Chosen per fold | Sharpe [90% CI] | Return "
        "| Max DD | Alpha/yr (t) | Beta | Gross Sharpe | Stress Sharpe | Folds Sharpe > 0 | Code |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        m = manifest[name]
        w = m.get("walk_forward")
        if not w:
            continue
        stale = "" if m["code_sha256"] == current else " (stale)"
        lo, hi = w["sharpe_90"]
        fs = [v for v in w["fold_sharpe"].values() if v is not None and not pd.isna(v)]
        ab = w["vs_hold"]
        lines.append(
            f"| `{name}` | {m['timeframe']} | ${m['capital']:,.0f} | {_fold_choices(m)} "
            f"| {_num(w['base']['sharpe'])} [{_num(lo)}, {_num(hi)}] | {_pct(w['base']['return'])} "
            f"| {_pct(w['base']['max_drawdown'])} | {_pct(ab['alpha'])} ({_num(ab['alpha_t'])}) "
            f"| {_num(ab['beta'])} | {_num(w['gross']['sharpe'])} | {_num(w['stress']['sharpe'])} "
            f"| {sum(v > 0 for v in fs)}/{len(fs)} | `{m['code_sha256'][:8]}`{stale} |")
    test_hold = next((manifest[n]["hold"]["test"] for n in names), {})
    lines += [
        "", "## Single split (continuity with earlier results)", "",
        f"Chosen on data before {TEST_START}, tested {TEST_START} to end of data. Buy and "
        f"hold: {_pct(test_hold.get('return'))}, Sharpe {_num(test_hold.get('sharpe'))}. "
        "Train Sharpe is in-sample.", "",
        "| Experiment | Chosen parameters | Test return | Test Sharpe | Max DD | Gross Sharpe "
        "| Stress Sharpe | Train Sharpe | Trades/yr per ticker | Time in market |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        m = manifest[name]
        te, tr = m["portfolio"]["test"], m["portfolio"]["train"]
        params = ", ".join(f"{k}={v}" for k, v in sorted(m["params"].items())) or "defaults"
        if m.get("selection_note"):
            params += f" ({m['selection_note']})"
        lines.append(
            f"| `{name}` | {params} | {_pct(te['return'])} | {_num(te['sharpe'])} "
            f"| {_pct(te['max_drawdown'])} | {_num(m['portfolio'].get('gross', {}).get('sharpe'))} "
            f"| {_num(m['portfolio'].get('stress', {}).get('sharpe'))} | {_num(tr['sharpe'])} "
            f"| {m['test_trades_per_year']:.0f} | {m['test_exposure']:.0%} |")
    lines += ["", "Files: `<experiment>.csv` (single split per ticker), `_folds.csv` "
              "(walk-forward per fold and ticker), `_grid.csv` (tuning scores per fold), "
              "`_equity.csv` (single-split daily equity), `_wf_returns.csv` (walk-forward "
              "daily portfolio returns), `manifest.json` (code and data fingerprints).", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="experiments to run (default: all)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--data-dir", default="data/scratch/tickers")
    p.add_argument("--tickers", nargs="*", help="default: algotrader.universe.UNIVERSE")
    p.add_argument("--workers", type=int, default=2)
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
        eq = res["test_equity"]
        years = ((eq.index[-1] - pd.Timestamp(TEST_START)).days / 365.25) if len(eq) else 0
        entry = {"description": exp.description, "strategy": exp.strategy.__name__,
                 "timeframe": exp.timeframe, "params": res["params"],
                 "fold_params": res["fold_params"], "fold_notes": res["fold_notes"],
                 "grid": {k: [str(v) for v in vs] for k, vs in exp.grid.items()},
                 "costs": vars(exp.costs), "sizer": vars(exp.sizer), "fit": exp.fit,
                 "capital": exp.capital, "selection_note": res["note"],
                 "split": {"test_start": TEST_START, "fold_starts": FOLD_STARTS,
                           "valid_months": VALID_MONTHS},
                 "portfolio": res["portfolio"], "hold": res["hold"],
                 "walk_forward": res["walk_forward"],
                 "test_trades_per_year": float(test["trades"].mean() / years) if years > 0 else 0.0,
                 "test_exposure": float(test["exposure"].mean()) if len(test) else 0.0,
                 "code_sha256": code_fingerprint(), "git": git_state(),
                 "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "data": data}
        write_results(exp, res, args.out, entry)
        write_summary(args.out)
        w = res["walk_forward"]
        print(f"{exp.name}: walk-forward Sharpe {w['base']['sharpe']:.2f} "
              f"(gross {w['gross']['sharpe']:.2f}, B&H {w['hold']['sharpe']:.2f}) | "
              f"single-split params {res['params']}", flush=True)


if __name__ == "__main__":
    main()
