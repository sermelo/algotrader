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
import multiprocessing
import re
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
from .momentum import TimeSeriesMomentumStrategy, TurnOfMonthStrategy
from .metrics import Metrics
from .ml import GradientBoostingStrategy
from .overlays import MarketRegime, VolTargeted
from .portfolio import (CrossSectionalML, CrossSectionalMomentum, PairsStrategy,
                        PortfolioBacktester, ShortTermReversal)
from .sizing import PositionSizer
from .stats import (TRADING_DAYS, alpha_beta, expected_max_sharpe, portfolio_returns,
                    returns_stats, sharpe_interval)
from .strategy import SmaCrossStrategy
from .tree import DecisionTreeStrategy
from .universe import UNIVERSES

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
MARKET = "SPY"                     # market series for the regime overlay
OVERLAY_KEYS = ("vol_target", "vol_lookback", "regime_days")
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

    def make(self, params: dict, market=None):
        """Build the strategy for `params`, wrapped in any overlays they ask for:
        `vol_target` (+ `vol_lookback`) -> VolTargeted, `regime_days` -> MarketRegime
        on `market` bars. None disables an overlay."""
        core = {k: v for k, v in params.items() if k not in OVERLAY_KEYS}
        strat = self.strategy(**core)
        if params.get("vol_target"):
            strat = VolTargeted(strat, params["vol_target"], params.get("vol_lookback") or 20)
        if params.get("regime_days"):
            strat = MarketRegime(strat, params["regime_days"])
            if market is not None:
                strat.set_market(market)
        return strat

    @property
    def needs_market(self) -> bool:
        return any(p.get("regime_days") for p in self.combos())

    def combos(self) -> list:
        """Parameter dicts to try: `params` updated with each point of `grid`."""
        keys = list(self.grid)
        out = []
        for values in itertools.product(*(self.grid[k] for k in keys)):
            p = {**self.params, **dict(zip(keys, values))}
            try:
                self.make(p)                         # skip invalid points (e.g. fast >= slow)
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


@dataclass(frozen=True)
class PortfolioExperiment(Experiment):
    """A cross-sectional strategy run as ONE portfolio over all tickers (daily bars).
    `capital` is the whole portfolio's starting equity. With `allow_short`, weights
    may be negative (gross exposure capped at `max_gross`, shorts pay `borrow_bps` a
    year)."""
    capital: float = 100_000.0
    allow_short: bool = False
    max_gross: float = 1.0
    borrow_bps: float = 50.0

    def make(self, params: dict, market=None):
        return self.strategy(**params)


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
    # round 3: new strategies
    Experiment("tsmom_daily", "Moskowitz et al. (2012) time-series momentum, long only",
               TimeSeriesMomentumStrategy, "1D", {},
               {"lookback": [63, 126, 252], "skip": [0, 21], "vol_target": [None, 0.15]}),
    Experiment("turn_of_month_daily", "Lakonishok & Smidt (1988) turn of the month",
               TurnOfMonthStrategy, "1D", {},
               {"days_before": [0, 1, 2, 3], "days_after": [1, 3, 5]}),
    # round 3: overlays and filters on the earlier strategies
    Experiment("sma_daily_overlays", "SMA crossover with volatility targeting / SPY regime",
               SmaCrossStrategy, "1D", {"atr_period": 14, "atr_mult": 50.0},
               {"fast": [10, 20, 50], "slow": [50, 100, 200], "vol_target": [None, 0.10, 0.20],
                "regime_days": [None, 200]}),
    Experiment("rsi2_daily_overlays", "RSI(2) with volatility targeting / SPY regime",
               RsiReversionStrategy, "1D", {"rsi_period": 2},
               {"entry": [5.0, 10.0, 20.0], "exit_sma": [5, 10], "trend_sma": [None, 200],
                "vol_target": [None, 0.15], "regime_days": [None, 200]}),
    Experiment("overnight_1min_100k_filters", "Overnight hold, $100k, SPY regime and own trend",
               OvernightStrategy, "1min", {},
               {"trend_days": [None, 200], "regime_days": [None, 200]}, capital=100_000.0),
    Experiment("orb_5min_100k_filters", "Opening-range breakout, $100k, trend and regime filters",
               OpeningRangeBreakoutStrategy, "5min", {},
               {"or_minutes": [5, 15, 30], "min_rel_volume": [0.0, 1.5], "trend_days": [None, 20],
                "regime_days": [None, 200]}, capital=100_000.0),
    # round 3: cross-sectional portfolios (one portfolio over all tickers, $100k)
    PortfolioExperiment("xs_momentum_daily", "Jegadeesh & Titman (1993) momentum, top N",
                        CrossSectionalMomentum, "1D", {"every": "M"},
                        {"lookback": [126, 252], "skip": [0, 21], "top_n": [5, 10]}),
    PortfolioExperiment("xs_reversal_daily", "Jegadeesh (1990) short-term reversal, bottom N",
                        ShortTermReversal, "1D", {},
                        {"lookback": [5, 21], "top_n": [5, 10], "every": ["W", "M"]}),
    # round 4: long-short (shorting, 1x gross, 50 bp/yr borrow)
    PortfolioExperiment("xs_momentum_ls_daily", "Momentum long top N, short bottom N",
                        CrossSectionalMomentum, "1D", {"every": "M"},
                        {"lookback": [126, 252], "skip": [0, 21], "top_n": [5, 10],
                         "short_n": [5, 10]}, allow_short=True),
    PortfolioExperiment("pairs_daily", "Pairs mean reversion on fixed economic pairs",
                        PairsStrategy, "1D", {},
                        {"lookback": [20, 60], "entry_z": [1.5, 2.0], "exit_z": [0.0, 0.5]},
                        allow_short=True),
    PortfolioExperiment("xs_gbm_daily", "Gradient boosting ranks tickers by forward return",
                        CrossSectionalML, "1D", {"every": "W"},
                        {"horizon": [5, 21], "top_n": [5, 10]}, fit="pooled"),
    PortfolioExperiment("xs_gbm_ls_daily", "Gradient-boosting ranking, long top N, short bottom N",
                        CrossSectionalML, "1D", {"every": "W"},
                        {"horizon": [5, 21], "top_n": [5, 10], "short_n": [5, 10]},
                        fit="pooled", allow_short=True),
]

# Later additions import their strategies here, inside the registry section, which is
# left out of the fingerprints: adding experiments then changes no other experiment's
# fingerprint. (Move the imports above here too the next time the runner changes.)
from .factors import DualMomentum, High52Week, LowVolatility, SectorMomentum  # noqa: E402

EXPERIMENTS += [
    PortfolioExperiment("low_vol_daily", "Ang et al. (2006) low volatility, lowest-vol N",
                        LowVolatility, "1D", {"every": "M"},
                        {"lookback": [63, 126], "top_n": [5, 10, 20]}),
    PortfolioExperiment("high52_daily", "George & Hwang (2004) 52-week-high momentum",
                        High52Week, "1D", {"every": "M"},
                        {"lookback": [126, 252], "top_n": [5, 10, 20]}),
    PortfolioExperiment("sector_momentum_daily", "Moskowitz & Grinblatt (1999) sector ETFs",
                        SectorMomentum, "1D", {"every": "M"},
                        {"lookback": [63, 126, 252], "top_n": [2, 3, 4]}),
    PortfolioExperiment("dual_momentum_daily", "Antonacci dual momentum (SPY/QQQ/IWM/GLD, TLT)",
                        DualMomentum, "1D", {"every": "M"},
                        {"lookback": [126, 252], "top_n": [1, 2], "safe": ["TLT", None]}),
]

# Roadmap S1 (buy/hold buffer, Novy-Marx & Velikov 2016) and S2 (multi-horizon trend,
# Hurst, Ooi & Pedersen 2017).
from .buffered import BufferedMomentum  # noqa: E402
from .trend import MultiHorizonTrend  # noqa: E402

EXPERIMENTS += [
    PortfolioExperiment("xs_momentum_buffer_daily", "Momentum top N with a keep-N buffer",
                        BufferedMomentum, "1D", {"every": "M"},
                        {"lookback": [126, 252], "skip": [0, 21], "top_n": [5, 10],
                         "keep_n": [10, 20]}),
    PortfolioExperiment("xs_momentum_buffer_ls_daily", "Long-short momentum with buffers",
                        BufferedMomentum, "1D", {"every": "M"},
                        {"lookback": [126, 252], "skip": [0, 21], "top_n": [5, 10],
                         "keep_n": [10, 20], "short_n": [5, 10]}, allow_short=True),
    PortfolioExperiment("trend_multi_daily", "1/3/12-month trend on ETFs, vol-scaled, long-short",
                        MultiHorizonTrend, "1D", {"every": "M"},
                        {"target_vol": [0.10, 0.20], "vol_lookback": [21, 63]}, allow_short=True),
    PortfolioExperiment("trend_multi_long_daily", "1/3/12-month trend on ETFs, long or cash",
                        MultiHorizonTrend, "1D", {"every": "M", "long_only": True},
                        {"target_vol": [0.10, 0.20], "vol_lookback": [21, 63]}),
]


# Roadmap S13: pre-holiday effect (Ariel 1990).
from .calendar_effects import PreHolidayStrategy  # noqa: E402

EXPERIMENTS += [
    Experiment("pre_holiday_daily", "Ariel (1990) pre-holiday effect", PreHolidayStrategy, "1D",
               {}, {"days_before": [1, 2], "days_after": [0, 1]}),
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


MAX_TRADE_ROWS = 300_000       # larger trade logs are not written (noted in the manifest)


def _window_record(result, bars, start, end, with_trades=False):
    """(summary, daily strategy curve, daily buy-and-hold curve, trades) for one
    window; trades is a DataFrame of the window's trades when `with_trades`."""
    part = slice_result(result, start, end)
    if part is None:
        return None
    period = bars[(bars.index >= start) if start is not None else slice(None)]
    period = _before(period, end)
    summary = Metrics(part, period).summary()
    trades = part.trades_frame() if with_trades else None
    return (summary, _daily_curve(part.equity, part.initial_capital),
            _daily_curve(period["close"], period["open"].iloc[0]), trades)


# ------------------------------------------------------------------- workers

def _market(exp, data_dir):
    """Market bars at the experiment's bar size, if any grid point needs them."""
    return BarLoader(data_dir).load(MARKET, exp.timeframe) if exp.needs_market else None


def _engine(exp, strat, variant):
    return Backtester(strat, exp.costs_for(variant), exp.sizer, exp.capital)


def _select_ticker(args):
    """Pass 1 worker: score every grid point at every selection date on one ticker."""
    exp_name, ticker, data_dir, starts, combos, models = args
    exp = BY_NAME[exp_name]
    bars = BarLoader(data_dir).load(ticker, exp.timeframe)
    market = _market(exp, data_dir)
    out = []
    if not exp.fit:
        for k, p in enumerate(combos):
            full = _engine(exp, exp.make(p, market), "base").run(bars)
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
                strat = exp.make(p, market)
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
    market = _market(exp, data_dir)
    cache, out = {}, []
    for name, start, end, params, fit_end in windows:
        variants = ("base",) if name == "train" else VARIANTS
        if not exp.fit:
            for var in variants:
                key = (json.dumps(params, sort_keys=True), var)
                if key not in cache:
                    cache[key] = _engine(exp, exp.make(params, market), var).run(bars)
                rec = _window_record(cache[key], bars, start, end,
                                     with_trades=var == "base" and name.startswith("fold:"))
                if rec:
                    out.append((name, var, ticker) + rec)
            continue
        if exp.fit == "pooled":
            strat = models[fit_end]
        else:
            strat = exp.make(params, market)
            try:
                strat.fit(bars[bars.index < fit_end])
            except ValueError:
                continue
        upto = _before(bars, end)
        if start is not None and not (upto.index >= start).any():
            continue
        for var in variants:
            run = _engine(exp, strat, var).run(upto, start=start)
            rec = _window_record(run, bars, start, end,
                                 with_trades=var == "base" and name.startswith("fold:"))
            if rec:
                out.append((name, var, ticker) + rec)
    return out


def _map(fn, jobs, workers):
    """Run jobs in `workers` processes. Workers are spawned, not forked: forking after
    scikit-learn's OpenMP threads have started in this process can deadlock them
    (it hung a pooled gradient-boosting run)."""
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(fn, jobs))


# ------------------------------------------------------------------- driver

def _train_frames(exp, data_dir, tickers, end) -> list:
    """Every ticker's bars before `end`, for pooled fitting."""
    loader = BarLoader(data_dir)
    return [_before(loader.load(t, exp.timeframe), end) for t in tickers]


def _fit_pooled(exp, params, frames, end, market=None):
    """Fit one strategy on all frames, using only bars before `end`."""
    return exp.make(params, market).fit_many([f[f.index < end] for f in frames])


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
    market = _market(exp, data_dir) if pooled else None

    # pass 1: selection
    chosen, notes, grid_rows = {F: 0 for F in starts}, {F: "" for F in starts}, []
    if len(combos) > 1:
        models = ({(F, k): _fit_pooled(exp, p, frames, valid_start(F), market)
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
    models = ({F: _fit_pooled(exp, combos[chosen[F]], frames, F, market) for F in starts}
              if pooled else None)
    del frames
    jobs = [(exp.name, t, str(data_dir), windows, models) for t in tickers]
    recs = [r for chunk in _map(_final_ticker, jobs, workers) for r in chunk]
    return _assemble(exp, combos, chosen, notes, grid_rows, recs)


def _assemble(exp, combos, chosen, notes, grid_rows, recs) -> dict:
    """Portfolio statistics for the single split and the walk-forward."""
    by, trade_frames = {}, []
    for name, var, ticker, summary, daily, hold, trades in recs:
        by.setdefault((name, var), {})[ticker] = (summary, daily, hold)
        if trades is not None and len(trades):
            trade_frames.append(trades.assign(ticker=ticker, fold=name.split(":")[1]))

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
        "trades": pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
    }


# ------------------------------------------------------------- portfolios

def _portfolio_summary(res, start=None, end=None):
    """(summary, daily curve) of a portfolio run inside [start, end)."""
    eq = res.equity
    mask = np.ones(len(eq), dtype=bool)
    if start is not None:
        mask &= eq.index >= start
    if end is not None:
        mask &= eq.index < end
    if not mask.any():
        return None
    before = eq[eq.index < start] if start is not None else eq.iloc[:0]
    initial = float(before.iloc[-1]) if len(before) else res.initial_capital
    part = eq[mask]
    orders = [o for o in res.orders
              if (start is None or o[0] >= start) and (end is None or o[0] < end)]
    curve = _daily_curve(part, initial)
    st = returns_stats(curve.pct_change().dropna())
    invested = (res.holdings[mask] > 0).any(axis=1).mean()
    summary = {"total_return": st["return"], "cagr": st["cagr"], "sharpe": st["sharpe"],
               "max_drawdown": st["max_drawdown"], "exposure": float(invested),
               "trades": len({o[0] for o in orders}),          # rebalance days
               "orders": len(orders), "total_commission": float(sum(o[4] for o in orders)),
               "final_equity": float(part.iloc[-1])}
    return summary, curve


def run_portfolio_experiment(exp: PortfolioExperiment, data_dir, tickers) -> dict:
    """Walk-forward for a cross-sectional strategy: one full-history run per grid
    point (and cost variant), sliced into selection and test windows."""
    loader = BarLoader(data_dir)
    frames = {t: loader.load(t, exp.timeframe) for t in tickers}
    combos, starts = exp.combos(), selection_starts()
    runs, models = {}, {}

    def strategy(k, fit_end):
        """Rule strategies are built once; fitted ones are fitted on bars before fit_end."""
        if not exp.fit:
            return exp.make(combos[k])
        if (k, fit_end) not in models:
            models[(k, fit_end)] = exp.make(combos[k]).fit_many(
                {t: f[f.index < fit_end] for t, f in frames.items()})
        return models[(k, fit_end)]

    def run(k, var, fit_end=None):
        key = (k, var, fit_end if exp.fit else None)
        if key not in runs:
            engine = PortfolioBacktester(strategy(k, fit_end), exp.costs_for(var), exp.capital,
                                         exp.allow_short, exp.max_gross,
                                         exp.borrow_bps if var != "gross" else 0.0)
            runs[key] = engine.run(frames)
        return runs[key]

    chosen, notes, grid_rows = {F: 0 for F in starts}, {F: "" for F in starts}, []
    if len(combos) > 1:
        for F in starts:
            rows = []
            for k, p in enumerate(combos):
                # fitted: fit before F - VALID_MONTHS, score on the validation months
                rec = (_portfolio_summary(run(k, "base", valid_start(F)), valid_start(F), F)
                       if exp.fit else _portfolio_summary(run(k, "base"), None, F))
                st, trades = (({"return": rec[0]["total_return"], "cagr": rec[0]["cagr"],
                                "sharpe": rec[0]["sharpe"], "max_drawdown": rec[0]["max_drawdown"]},
                               rec[0]["trades"]) if rec else ({}, 0))
                rows.append({"fold": F.date().isoformat(), "params": json.dumps(p, sort_keys=True),
                             **{f"portfolio_{s}": st.get(s, np.nan)
                                for s in ("return", "cagr", "sharpe", "max_drawdown")},
                             "median_trades": float(trades), "eligible": trades >= MIN_TRADES})
            chosen[F], notes[F] = _choose(rows)
            grid_rows += rows

    hold = {t: f for t, f in frames.items()}

    def hold_returns(start, end):
        curves = {}
        for t, f in hold.items():
            per = _before(f[f.index >= start] if start is not None else f, end)
            if len(per):
                curves[t] = _daily_curve(per["close"], per["open"].iloc[0])
        return portfolio_returns(curves) if curves else pd.Series(dtype=float)

    single = pd.Timestamp(TEST_START)
    recs = {}
    windows = [(f"fold:{s.date()}", s, e, chosen[s], s) for s, e in fold_windows()]
    windows += [("test", single, None, chosen[single], single),
                ("train", None, single, chosen[single], single)]
    order_frames = []
    for name, start, end, k, fit_end in windows:
        for var in (("base",) if name == "train" else VARIANTS):
            res_run = run(k, var, fit_end)
            rec = _portfolio_summary(res_run, start, end)
            if rec:
                recs[(name, var)] = rec
            if var == "base" and name.startswith("fold:"):
                rows = [o for o in res_run.orders
                        if o[0] >= start and (end is None or o[0] < end)]
                if rows:
                    order_frames.append(pd.DataFrame(rows, columns=[
                        "date", "ticker", "shares", "price", "commission"]).assign(
                        fold=name.split(":")[1]))

    def rets(name, var):
        return recs[(name, var)][1].pct_change().dropna() if (name, var) in recs \
            else pd.Series(dtype=float)

    folds = [f"fold:{s.date()}" for s, _ in fold_windows()]
    wf = {var: pd.concat([rets(f, var) for f in folds]).sort_index() for var in VARIANTS}
    wf_hold = pd.concat([hold_returns(s, e) for s, e in fold_windows()]).sort_index()
    period = {("test", "base"): "test", ("test", "gross"): "gross", ("test", "stress"): "stress",
              ("train", "base"): "train"}
    rows = [{"ticker": "PORTFOLIO", "period": period.get(key, f"{key[0]}/{key[1]}"), **rec[0]}
            for key, rec in recs.items()]
    rows = pd.DataFrame(rows)
    single_rows = rows[rows["period"].isin(["train", "test", "gross", "stress"])]
    test_eq = recs.get(("test", "base"))
    return {
        "params": combos[chosen[single]], "note": notes[single],
        "fold_params": {s.date().isoformat(): combos[chosen[s]] for s, _ in fold_windows()},
        "fold_notes": {s.date().isoformat(): notes[s] for s, _ in fold_windows() if notes[s]},
        "rows": single_rows, "fold_rows": rows[~rows.index.isin(single_rows.index)],
        "grid": pd.DataFrame(grid_rows),
        "portfolio": {ph: returns_stats(rets(*key)) for ph, key in
                      (("train", ("train", "base")), ("test", ("test", "base")),
                       ("gross", ("test", "gross")), ("stress", ("test", "stress")))},
        "hold": {"train": returns_stats(hold_returns(None, single)),
                 "test": returns_stats(hold_returns(single, None))},
        "walk_forward": {
            **{var: returns_stats(wf[var]) for var in VARIANTS},
            "hold": returns_stats(wf_hold), "sharpe_90": sharpe_interval(wf["base"]),
            "vs_hold": alpha_beta(wf["base"], wf_hold),
            "fold_sharpe": {f.split(":")[1]: returns_stats(rets(f, "base"))["sharpe"]
                            for f in folds},
            "days": int(len(wf["base"]))},
        "wf_returns": pd.DataFrame({**{var: wf[var] for var in VARIANTS}, "hold": wf_hold}),
        "test_equity": (pd.DataFrame({"PORTFOLIO": test_eq[1]}) if test_eq
                        else pd.DataFrame()),
        "trades": pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame(),
    }


# --------------------------------------------------------------- fingerprints

def code_fingerprint() -> str:
    """sha256 over all package source files (name + content), first 16 hex digits."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


# modules every experiment's numbers depend on (besides its strategy's own modules)
CORE_MODULES = ("engine", "costs", "sizing", "data", "metrics", "stats", "indicators",
                "strategy", "overlays")


def _module_deps(name: str, seen: set) -> set:
    """`name` plus the package modules it imports (`from .x import`), recursively."""
    if name in seen:
        return seen
    seen.add(name)
    src = (PACKAGE_DIR / f"{name}.py").read_text()
    for dep in re.findall(r"^from \.(\w+) import", src, flags=re.M):
        _module_deps(dep, seen)
    return seen


def runner_source() -> str:
    """experiments.py without the EXPERIMENTS registry, so adding or editing one
    experiment does not change the others' fingerprints."""
    src = (PACKAGE_DIR / "experiments.py").read_text()
    start, end = src.index("\nEXPERIMENTS = ["), src.index("\nBY_NAME = ")
    return src[:start] + src[end:]


def experiment_fingerprint(exp: Experiment) -> str:
    """sha256 over the code one experiment's results depend on: the runner (without
    the registry), the engine/cost/metric modules, the strategy's module and the
    package modules it imports, the portfolio engine for portfolio experiments, and the
    experiment's own definition. First 16 hex digits."""
    modules = set()
    for m in CORE_MODULES + ((("portfolio",) if isinstance(exp, PortfolioExperiment) else ())):
        _module_deps(m, modules)
    _module_deps(exp.strategy.__module__.rsplit(".", 1)[-1], modules)
    h = hashlib.sha256(runner_source().encode())
    for m in sorted(modules):
        h.update(m.encode())
        h.update((PACKAGE_DIR / f"{m}.py").read_bytes())
    h.update(repr(exp).encode())
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
    trades = res.get("trades", pd.DataFrame())
    manifest_entry["trade_log"] = {"rows": int(len(trades)),
                                   "written": bool(0 < len(trades) <= MAX_TRADE_ROWS)}
    path = out / f"{exp.name}_trades.csv.gz"
    if manifest_entry["trade_log"]["written"]:
        trades.to_csv(path, index=False, float_format="%.6g", compression="gzip")
    elif path.exists():
        path.unlink()
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
        stale = "" if m["code_sha256"] == experiment_fingerprint(BY_NAME[name]) else " (stale)"
        lo, hi = w["sharpe_90"]
        fs = [v for v in w["fold_sharpe"].values() if v is not None and not pd.isna(v)]
        ab = w["vs_hold"]
        lines.append(
            f"| `{name}` | {m['timeframe']} | ${m['capital']:,.0f}"
            f"{' total' if m.get('kind') == 'portfolio' else ''} | {_fold_choices(m)} "
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
    p.add_argument("--universe", choices=sorted(UNIVERSES), default="core",
                   help="ticker set from algotrader.universe (default core)")
    p.add_argument("--tickers", nargs="*", help="explicit tickers (overrides --universe)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", help="default: results/ for core, results/<universe>/ otherwise")
    args = p.parse_args(argv)
    if args.list:
        for e in EXPERIMENTS:
            print(f"{e.name:24s} {e.timeframe:6s} {len(e.combos()):3d} combos  {e.description}")
        return
    unknown = [n for n in args.names if n not in BY_NAME]
    if unknown:
        p.error(f"unknown experiments: {unknown}")
    tickers = args.tickers or UNIVERSES[args.universe]
    out = args.out or ("results" if args.universe == "core" else f"results/{args.universe}")
    data = data_fingerprint(args.data_dir, tickers)
    for exp in [BY_NAME[n] for n in args.names] if args.names else EXPERIMENTS:
        res = (run_portfolio_experiment(exp, args.data_dir, tickers)
               if isinstance(exp, PortfolioExperiment)
               else run_experiment(exp, args.data_dir, tickers, args.workers))
        test = res["rows"][res["rows"]["period"] == "test"]
        eq = res["test_equity"]
        years = ((eq.index[-1] - pd.Timestamp(TEST_START)).days / 365.25) if len(eq) else 0
        entry = {"description": exp.description, "strategy": exp.strategy.__name__,
                 "timeframe": exp.timeframe, "params": res["params"],
                 "fold_params": res["fold_params"], "fold_notes": res["fold_notes"],
                 "grid": {k: [str(v) for v in vs] for k, vs in exp.grid.items()},
                 "costs": vars(exp.costs), "sizer": vars(exp.sizer), "fit": exp.fit,
                 "capital": exp.capital, "selection_note": res["note"],
                 "universe": "custom" if args.tickers else args.universe,
                 "kind": "portfolio" if isinstance(exp, PortfolioExperiment) else "per ticker",
                 "tickers": len(tickers),
                 "split": {"test_start": TEST_START, "fold_starts": FOLD_STARTS,
                           "valid_months": VALID_MONTHS},
                 "portfolio": res["portfolio"], "hold": res["hold"],
                 "walk_forward": res["walk_forward"],
                 "test_trades_per_year": float(test["trades"].mean() / years) if years > 0 else 0.0,
                 "test_exposure": float(test["exposure"].mean()) if len(test) else 0.0,
                 "code_sha256": experiment_fingerprint(exp),
                 "package_sha256": code_fingerprint(), "git": git_state(),
                 "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "data": data}
        write_results(exp, res, out, entry)
        write_summary(out)
        w = res["walk_forward"]
        print(f"{exp.name}: walk-forward Sharpe {w['base']['sharpe']:.2f} "
              f"(gross {w['gross']['sharpe']:.2f}, B&H {w['hold']['sharpe']:.2f}) | "
              f"single-split params {res['params']}", flush=True)


if __name__ == "__main__":
    main()
