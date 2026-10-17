"""Walk-forward of portfolio strategies on the long point-in-time S&P 500 history.

Separate from `experiments.py` (whose folds cover 2023-2026 on 1-minute data), so its
results and fingerprints are independent. Universe: on each day, only the stocks that
were S&P 500 members that day can be bought (`PointInTime`); a stock that leaves the
index is sold at the next rebalance, and one whose data ends is sold at its last close.
Prices come from `data/history` (Tiingo where its series passed `history check`, else
the adjusted Hugging Face series, else Quandl WIKI). Folds start each January from `FIRST_FOLD`; at each fold start the
parameters with the best portfolio Sharpe on all earlier data are chosen.

Example:
    python -m algotrader.longrun --list
    python -m algotrader.longrun xs_momentum_long        # -> results/long/
"""
import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .costs import CostModel
from .history import member_spans, parse_members, tiingo_symbol, wiki_symbol
from .portfolio import PortfolioBacktester
from .stats import alpha_beta, expected_max_sharpe, returns_stats, sharpe_interval

HISTORY = Path("data/history")
OUT = Path("results/long")
START = "2008-01-01"
FIRST_FOLD = 2012
CAPITAL = 1_000_000.0
COSTS = {"base": CostModel(spread_bps=2.0), "gross": CostModel(0.0, 0.0, 0.0, 0.0),
         "stress": CostModel(spread_bps=5.0)}
BORROW_BPS = 50.0
BENCHMARK = "SPY"
MIN_REBALANCES = 3            # a grid point must trade on at least this many days


# ---------------------------------------------------------------- universe

@dataclass(frozen=True)
class PointInTime:
    """Wraps a cross-sectional strategy so it only sees members: closes of tickers that
    are not members on a day are hidden (NaN) when weights are computed."""
    inner: object
    members: pd.DataFrame = field(repr=False, compare=False)   # bool, dates x tickers

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        mask = self.members.reindex(columns=closes.columns, fill_value=False).astype(float)
        mask = mask.reindex(index=closes.index.union(mask.index)).ffill()  # lists are not daily
        mask = mask.reindex(closes.index).fillna(0.0) > 0.5
        return self.inner.weights(closes.where(mask))


def membership_matrix(members: pd.DataFrame) -> pd.DataFrame:
    """Bool DataFrame (membership-list dates x tickers) from the long (date, ticker) table."""
    m = members.assign(v=True).pivot_table(index="date", columns="ticker", values="v",
                                           aggfunc="any", fill_value=False)
    return m.astype(bool)


def load_frames(history=HISTORY, start=START, coverage=None, use_hf=True) -> dict:
    """Daily frames per member ticker (and the benchmark): Tiingo unless coverage.csv
    marks it a mismatch or covering under half of the membership period (a reused
    ticker), else the adjusted Hugging Face series in `history/hf` (`hfdata.build_source`,
    skipped with `use_hf=False`), else WIKI. Columns open..volume (+ raw_close)."""
    history = Path(history)
    spans = member_spans(parse_members((history / "sp500_members.csv").read_text()), start)
    cov = None
    path = Path(coverage) if coverage else history / "coverage.csv"
    if path.exists():
        cov = pd.read_csv(path, index_col=0)
    frames = {}
    for t in list(spans.index) + [BENCHMARK]:
        tiingo = history / "tiingo" / f"{tiingo_symbol(t)}.parquet"
        wiki = history / "wiki" / f"{wiki_symbol(t)}.parquet"
        # wrong company (returns disagree with WIKI/HF) or mostly outside the membership
        bad = cov is not None and t in cov.index and (
            cov.loc[t, "status"] == "mismatch"
            or (cov.loc[t, "status"] == "partial" and not cov.loc[t].get("cover", 1) >= 0.5))
        hf = history / "hf" / f"{wiki_symbol(t)}.parquet"
        src = (tiingo if tiingo.exists() and not bad else hf if use_hf and hf.exists()
               else wiki if wiki.exists() else None)
        if src is None:
            continue
        df = pd.read_parquet(src)
        df = df[[c for c in ("open", "high", "low", "close", "volume", "raw_close") if c in df]]
        df = df[df.index >= pd.Timestamp(start)].dropna(subset=["open", "close"])
        if len(df):
            frames[t] = df
    return frames


# ---------------------------------------------------------------- experiments

@dataclass(frozen=True)
class LongExperiment:
    name: str
    description: str
    strategy: type
    fixed: dict
    grid: dict
    allow_short: bool = False
    prepare: object = field(default=None, repr=False, compare=False)   # (frames, members) -> kwargs
    first_fold: int = FIRST_FOLD

    def combos(self) -> list:
        keys = sorted(self.grid)
        out = [{}]
        for k in keys:
            out = [c | {k: v} for c in out for v in self.grid[k]]
        valid = []
        for c in out:
            try:
                self.strategy(**self.fixed, **c)
                valid.append(c)
            except ValueError:
                pass
        return valid

    def make(self, params, extra=None):
        return self.strategy(**self.fixed, **(extra or {}), **params)


EDGAR = Path("data/edgar")


_PANELS = {}


def fundamental_panels(frames: dict, members=None, edgar=EDGAR) -> dict:
    """Signal panels for FundamentalSort from the SEC facts and unadjusted closes
    (computed once per set of frames)."""
    from .fundamentals import signals
    if id(frames) in _PANELS:
        return _PANELS[id(frames)]
    raw = pd.DataFrame({t: f["raw_close"] for t, f in frames.items() if "raw_close" in f})
    facts = pd.read_parquet(Path(edgar) / "facts.parquet")
    cik = pd.read_csv(Path(edgar) / "cik_map.csv", index_col=0)
    _PANELS[id(frames)] = {"panels": signals(facts, cik, raw)}
    return _PANELS[id(frames)]


def earnings_panels(frames: dict, members=None, edgar=EDGAR) -> dict:
    """EAR panels (hold 21 and 63 days) for EarningsDrift from the SEC earnings releases;
    the market is SPY if loaded, else the equal-weight average of the stocks."""
    from .pead import announcement_returns
    key = ("ear", id(frames))
    if key in _PANELS:
        return _PANELS[key]
    closes = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    if BENCHMARK in closes:
        market = closes[BENCHMARK]
    else:
        market = (1 + closes.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()
    earn = pd.read_parquet(Path(edgar) / "earnings.parquet")
    cik = pd.read_csv(Path(edgar) / "cik_map.csv", index_col=0)
    stocks = closes.drop(columns=[BENCHMARK], errors="ignore")
    _PANELS[key] = {"panels": {f"ear_{h}": announcement_returns(earn, cik, stocks, market, h)
                               for h in (21, 63)}}
    return _PANELS[key]


DEEP_EPOCHS = 100


def deep_arb_panels(frames: dict, members=None, k: int = 5, epochs=None) -> dict:
    """Weights of the deep statistical-arbitrage model (PCA residuals of that day's
    members, walk-forward refits); slow: run on the VPS."""
    from .deep_arb import DeepArbModel, pca_residuals
    key = ("deep", id(frames), k)
    if key in _PANELS:
        return _PANELS[key]
    closes = pd.DataFrame({t: f["close"] for t, f in frames.items() if t != BENCHMARK})
    rets = closes.pct_change(fill_method=None)
    daily = None
    if members is not None:
        m = members.reindex(columns=closes.columns, fill_value=False).astype(float)
        daily = m.reindex(index=closes.index.union(m.index)).ffill().reindex(closes.index).fillna(0.0) > 0.5
    eps, load = pca_residuals(rets, daily, k=k)
    model = DeepArbModel(k=k, epochs=epochs or DEEP_EPOCHS)
    w = model.weights_panel(rets, eps, load)
    Path("data/long_panels").mkdir(parents=True, exist_ok=True)       # for diagnostics
    w.to_parquet(f"data/long_panels/deep_arb_k{k}.parquet")
    pd.DataFrame(model.log, columns=["fit_end", "epoch", "train_sharpe"]).to_csv(
        f"data/long_panels/deep_arb_k{k}_training.csv", index=False)
    _PANELS[key] = {"panels": {f"deep_arb_k{k}": w}}
    return _PANELS[key]


def _experiments():
    from .buffered import BufferedMomentum
    from .deep_arb import DeepArb
    from .pead import EarningsDrift
    from .seasonal import SameMonthSeasonality
    from .fundamentals import FundamentalSort
    from .factors import High52Week, LowVolatility
    from .portfolio import CrossSectionalMomentum, ShortTermReversal

    mom = {"lookback": [126, 252], "skip": [0, 21], "top_n": [20, 50]}
    exps = [
        LongExperiment("xs_momentum_long", "Jegadeesh & Titman momentum, top N members",
                       CrossSectionalMomentum, {"every": "M"}, mom),
        LongExperiment("xs_momentum_ls_long", "Momentum long top N, short bottom N",
                       CrossSectionalMomentum, {"every": "M"}, mom | {"short_n": [20, 50]},
                       allow_short=True),
        LongExperiment("xs_momentum_buffer_long", "Momentum with a keep-2N buffer",
                       BufferedMomentum, {"every": "M"},
                       {"lookback": [126, 252], "skip": [0, 21], "top_n": [20, 50],
                        "keep_n": [40, 100]}),
        LongExperiment("xs_reversal_long", "Short-term reversal, bottom N",
                       ShortTermReversal, {}, {"lookback": [5, 21], "top_n": [20, 50],
                                               "every": ["W", "M"]}),
        LongExperiment("low_vol_long", "Low volatility, lowest-vol N",
                       LowVolatility, {"every": "M"}, {"lookback": [63, 252], "top_n": [20, 50, 100]}),
        LongExperiment("high52_long", "52-week-high momentum",
                       High52Week, {"every": "M"}, {"lookback": [126, 252], "top_n": [20, 50, 100]}),
    ]
    # Roadmap S3: point-in-time fundamentals (Fama & French 1992; Novy-Marx 2013;
    # Cooper, Gulen & Schill 2008), long-only and long-short.
    for name, sig, largest, desc in (
            ("value", "book_to_market", True, "high book-to-market"),
            ("profitability", "gross_profitability", True, "high gross profits / assets"),
            ("asset_growth", "asset_growth", False, "low total-asset growth"),
            ("op_profitability", "operating_profitability", True,
             "high operating income / book equity (RMW-style)")):
        fixed = {"signal": sig, "largest": largest, "every": "M"}
        exps.append(LongExperiment(f"{name}_long", f"Long {desc}", FundamentalSort, fixed,
                                   {"top_n": [20, 50, 100]}, prepare=fundamental_panels))
        exps.append(LongExperiment(f"{name}_ls_long", f"Long {desc}, short the opposite",
                                   FundamentalSort, fixed, {"top_n": [20, 50], "short_n": [20, 50]},
                                   allow_short=True, prepare=fundamental_panels))
    # Roadmap S4: same-calendar-month seasonality (Heston & Sadka 2008).
    exps.append(LongExperiment("seasonality_long", "Long the best past same-month returns",
                               SameMonthSeasonality, {"years": 10},
                               {"min_years": [3, 5], "top_n": [20, 50]}))
    exps.append(LongExperiment("seasonality_ls_long", "Long best, short worst same-month returns",
                               SameMonthSeasonality, {"years": 10},
                               {"min_years": [3, 5], "top_n": [20, 50], "short_n": [20, 50]},
                               allow_short=True))
    # Roadmap S8: deep learning statistical arbitrage (Guijarro-Ordonez, Pelger & Zanotti);
    # first model after 252 + 1,000 days of data, so its folds start in 2013.
    exps.append(LongExperiment("deep_arb_long", "CNN+transformer on 5-factor PCA residuals",
                               DeepArb, {"k": 5}, {}, allow_short=True,
                               prepare=deep_arb_panels, first_fold=2013))
    # Roadmap S7: post-earnings drift on the announcement return (Brandt et al. 2008).
    exps.append(LongExperiment("pead_long", "Long the highest recent earnings-announcement returns",
                               EarningsDrift, {"every": "M"},
                               {"top_n": [20, 50], "hold_days": [21, 63]}, prepare=earnings_panels))
    exps.append(LongExperiment("pead_ls_long", "Long highest, short lowest announcement returns",
                               EarningsDrift, {"every": "M", "hold_days": 63},
                               {"top_n": [20, 50], "short_n": [20, 50]}, allow_short=True,
                               prepare=earnings_panels))
    return exps


EXPERIMENTS = _experiments()
BY_NAME = {e.name: e for e in EXPERIMENTS}


# ---------------------------------------------------------------- walk-forward

def _returns(equity: pd.Series, start=None, end=None) -> pd.Series:
    r = equity.pct_change()
    if start is not None:
        r = r[r.index >= start]
    if end is not None:
        r = r[r.index < end]
    return r.dropna()


def _rebalances(res, start=None, end=None) -> int:
    return len({o[0] for o in res.orders if (start is None or o[0] >= start)
                and (end is None or o[0] < end)})


def turnover(res, start=None, end=None) -> float:
    """One-sided annual turnover: traded value / 2 / mean equity / years."""
    orders = [o for o in res.orders if (start is None or o[0] >= start) and (end is None or o[0] < end)]
    eq = res.equity
    eq = eq[(eq.index >= start) if start is not None else slice(None)]
    if end is not None:
        eq = eq[eq.index < end]
    if not len(eq) or not orders:
        return 0.0
    years = max(len(eq) / 252, 1e-9)
    traded = sum(abs(o[2]) * o[3] for o in orders)
    return float(traded / 2 / eq.mean() / years)


def fold_starts(index: pd.DatetimeIndex, first=FIRST_FOLD) -> list:
    years = range(first, index.max().year + 1)
    return [pd.Timestamp(f"{y}-01-01") for y in years if pd.Timestamp(f"{y}-01-01") < index.max()]


def run_long(exp: LongExperiment, frames: dict, members: pd.DataFrame) -> dict:
    combos = exp.combos()
    extra = exp.prepare(frames, members) if exp.prepare else None
    runs = {}

    def run(k, var):
        if (k, var) not in runs:
            strat = PointInTime(exp.make(combos[k], extra), members)
            engine = PortfolioBacktester(strat, COSTS[var], CAPITAL, exp.allow_short, 1.0,
                                         BORROW_BPS if (exp.allow_short and var != "gross") else 0.0)
            runs[(k, var)] = engine.run(frames)
        return runs[(k, var)]

    index = next(iter(frames.values())).index
    for f in frames.values():
        index = index.union(f.index)
    starts = fold_starts(index, exp.first_fold)
    grid, chosen = [], {}
    for F in starts:
        best, best_s = 0, -np.inf
        for k, p in enumerate(combos):
            res = run(k, "base")
            r = _returns(res.equity, None, F)
            s = returns_stats(r)["sharpe"] if len(r) > 20 else np.nan
            ok = _rebalances(res, None, F) >= MIN_REBALANCES
            grid.append({"fold": F.date().isoformat(), "params": json.dumps(p, sort_keys=True),
                         "sharpe": s, "eligible": ok})
            if ok and s == s and s > best_s:
                best, best_s = k, s
        chosen[F] = best
    ends = starts[1:] + [None]
    wf = {var: [] for var in COSTS}
    folds = []
    for F, E in zip(starts, ends):
        k = chosen[F]
        for var in COSTS:
            wf[var].append(_returns(run(k, var).equity, F, E))
        r = wf["base"][-1]
        folds.append({"fold": F.date().isoformat(), "params": json.dumps(combos[k], sort_keys=True),
                      "sharpe": returns_stats(r)["sharpe"] if len(r) > 1 else np.nan,
                      "return": returns_stats(r)["return"] if len(r) else np.nan,
                      "turnover": turnover(run(k, "base"), F, E)})
    wf = {var: pd.concat(v) for var, v in wf.items()}
    bench = frames.get(BENCHMARK)
    hold = bench["close"].pct_change().reindex(wf["base"].index) if bench is not None else None
    return {"exp": exp, "combos": combos, "chosen": chosen, "grid": pd.DataFrame(grid),
            "folds": pd.DataFrame(folds), "wf": pd.DataFrame(wf | ({"hold": hold} if hold is not None else {}))}


# ---------------------------------------------------------------- output

def fingerprint(exp: LongExperiment) -> str:
    import sys
    h = hashlib.sha256(Path(__file__).read_bytes())
    for mod in ("portfolio", "costs", "stats", "history", "fundamentals", "pead", "deep_arb",
                exp.strategy.__module__.rsplit(".", 1)[-1]):
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    h.update(repr((exp.name, exp.strategy.__name__, exp.fixed, exp.grid, exp.allow_short,
                   exp.first_fold)).encode())
    return h.hexdigest()


def data_fingerprint(history=HISTORY) -> str:
    h = hashlib.sha256()
    files = sorted(Path(history).glob("*/*.parquet")) + [Path(history) / "sp500_members.csv"]
    files += [EDGAR / "facts.parquet", EDGAR / "cik_map.csv", EDGAR / "earnings.parquet"]
    for p in files:
        if p.exists():
            h.update(f"{p.name}:{p.parent.name}:{p.stat().st_size}".encode())
    return h.hexdigest()


def write(res: dict, out=OUT, history=HISTORY):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    exp = res["exp"]
    res["wf"].to_csv(out / f"{exp.name}_wf_returns.csv", float_format="%.6g")
    res["folds"].to_csv(out / f"{exp.name}_folds.csv", index=False, float_format="%.6g")
    res["grid"].to_csv(out / f"{exp.name}_grid.csv", index=False, float_format="%.6g")
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[exp.name] = {"code_sha256": fingerprint(exp), "data_sha256": data_fingerprint(history),
                     "configurations": len(res["combos"])}
    man_path.write_text(json.dumps(man, indent=2, sort_keys=True))


def write_summary(out=OUT):
    out = Path(out)
    man = json.loads((out / "manifest.json").read_text())
    n_conf = sum(v["configurations"] for v in man.values())
    rows, years = [], 0.0
    for name in sorted(man):
        wf = pd.read_csv(out / f"{name}_wf_returns.csv", index_col=0, parse_dates=True)
        folds = pd.read_csv(out / f"{name}_folds.csv")
        r = wf["base"].dropna()
        years = max(years, len(r) / 252)
        st = returns_stats(r)
        lo, hi = sharpe_interval(r)
        ab = alpha_beta(r, wf["hold"]) if "hold" in wf else {}
        stale = name in BY_NAME and man[name]["code_sha256"] != fingerprint(BY_NAME[name])
        rows.append(f"| `{name}`{' (stale)' if stale else ''} | {st['sharpe']:.2f} [{lo:.2f}, {hi:.2f}] | "
                    f"{st['cagr']:+.1%} | {st['max_drawdown']:.1%} | "
                    f"{ab.get('alpha', np.nan):+.1%} ({ab.get('alpha_t', np.nan):.2f}) | {ab.get('beta', np.nan):.2f} | "
                    f"{returns_stats(wf['gross'].dropna())['sharpe']:.2f} | {returns_stats(wf['stress'].dropna())['sharpe']:.2f} | "
                    f"{folds['turnover'].mean():.0%} | {(folds['sharpe'] > 0).sum()}/{len(folds)} |")
    hold = None
    for name in sorted(man):
        wf = pd.read_csv(out / f"{name}_wf_returns.csv", index_col=0, parse_dates=True)
        if "hold" in wf:
            hold = wf["hold"].dropna()
            break
    hs = returns_stats(hold) if hold is not None and len(hold) else None
    text = f"""# Long-history results (point-in-time S&P 500)

Generated by `python -m algotrader.longrun`; do not edit by hand. Universe: stocks that
were S&P 500 members on each day; walk-forward with folds starting each January from
{FIRST_FOLD}, parameters chosen on all earlier data by portfolio Sharpe; ${CAPITAL:,.0f}
portfolio; costs IBKR + 2 bp half-spread (base), none (gross), 5 bp (stress); shorts
pay {BORROW_BPS:.0f} bp a year. Alpha/beta against {BENCHMARK} buy and hold
({'Sharpe %.2f, CAGR %+.1f%%' % (hs['sharpe'], 100 * hs['cagr']) if hs else 'n/a'} over the same days).
Turnover: one-sided, per year. {n_conf} configurations tried here; the best of that many
zero-edge strategies would show a Sharpe of about {expected_max_sharpe(n_conf, years):.2f}
by luck over {years:.1f} years.

| Experiment | Sharpe [90% CI] | CAGR | Max DD | Alpha/yr (t) | Beta | Gross Sharpe | Stress Sharpe | Turnover | Folds > 0 |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + "\n"
    (out / "summary.md").write_text(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--history", default=str(HISTORY))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--no-hf", action="store_true", help="ignore the Hugging Face series")
    args = p.parse_args(argv)
    if args.list:
        for e in EXPERIMENTS:
            print(f"{e.name:28s} {len(e.combos()):3d} configs  {e.description}")
        return
    names = args.names or [e.name for e in EXPERIMENTS]
    frames = load_frames(args.history, use_hf=not args.no_hf)
    members = membership_matrix(parse_members((Path(args.history) / "sp500_members.csv").read_text()))
    print(f"{len(frames)} tickers, {min(f.index.min() for f in frames.values()).date()} to "
          f"{max(f.index.max() for f in frames.values()).date()}")
    for n in names:
        res = run_long(BY_NAME[n], frames, members)
        write(res, args.out, args.history)
        r = res["wf"]["base"]
        print(f"{n}: walk-forward Sharpe {returns_stats(r)['sharpe']:.2f}, "
              f"{len(res['folds'])} folds")
    write_summary(args.out)


if __name__ == "__main__":
    main()
