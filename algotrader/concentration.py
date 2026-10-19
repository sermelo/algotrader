"""Concentration test: hold the top 3, 5, 10, 20 or 50 of a strategy, and combine strategies.

Question (sergio, 2026-10-19): would putting the money in the very top names (3 or 5) beat spreading
it over 20-50, and would the top 5 of each of the best five strategies beat one strategy?

On the long point-in-time S&P 500 history, eleven long-only strategies at the parameters their
walk-forward chose most often (`results/long/*_folds.csv`), each held at N = 3, 5, 10, 20, 50 (equal
weight, monthly, $1M, base and stress costs), plus rank bands 4-5, 6-10, 11-20 and 21-50 (do the very top
ranks earn more than the next ones?). Multi-strategy portfolios (equal parts of each strategy's
daily return, so the parts are rebalanced daily and shared names are not netted; costs are each
part's own): all eleven at top 5 or top 20, and the five strategies with the best Sharpe on all earlier
data at top 5 or top 20, chosen every January (and the single best, for comparison).

Pre-registered rule: concentration is better for a strategy if top 3 or top 5 minus top 50 (daily
returns) has a Fama-French alpha with t >= 2 at base costs; better in general only if that holds for
most strategies. The test period is 2012-01-01 on (as the other long results); the backtests start in
2008 so the yearly strategy choice has earlier data.

    python -m algotrader.concentration run --out data/long_conc     # backtests (VPS) -> returns.csv
    python -m algotrader.concentration report --out results/long_conc   # tables -> summary.md
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import longrun
from .longrun import LongExperiment, PointInTime
from .portfolio import PortfolioBacktester
from .stats import alpha_beta, returns_stats, sharpe_interval

SIZES = (3, 5, 10, 20, 50)
BANDS = ((4, 5), (6, 10), (11, 20), (21, 50))
BUFFERS = ((3, 10), (5, 20), (20, 50))           # buy the top N, sell only when out of the top K
TEST_START = pd.Timestamp(f"{longrun.FIRST_FOLD}-01-01")
COST_VARIANTS = ("base", "stress")


def strategies() -> dict:
    """{name: LongExperiment} with everything but `top_n` fixed."""
    from .combo import CombinedSignal
    from .factors import High52Week, LowVolatility
    from .fundamentals import FundamentalSort
    from .longrun_combo import combo_panels, market_panels
    from .pead import EarningsDrift
    from .portfolio import CrossSectionalMomentum
    from .seasonal import SameMonthSeasonality
    grid = {"top_n": list(SIZES)}
    out = [
        LongExperiment("momentum", "12-1 month momentum", CrossSectionalMomentum,
                       {"every": "M", "lookback": 252, "skip": 21}, grid),
        LongExperiment("low_vol", "lowest 252-day volatility", LowVolatility, {"every": "M", "lookback": 252}, grid),
        LongExperiment("high52", "nearest the 52-week high", High52Week, {"every": "M", "lookback": 252}, grid),
        LongExperiment("combo_price", "momentum + low vol + 52-week high", CombinedSignal,
                       {"every": "M", "components": ("momentum", "low_vol", "high52"), "min_components": 2},
                       grid, prepare=market_panels),
        LongExperiment("combo", "combined signal with profitability", CombinedSignal, {"every": "M"}, grid,
                       prepare=combo_panels),
        LongExperiment("pead", "earnings-announcement return", EarningsDrift, {"every": "M", "hold_days": 21},
                       grid, prepare=longrun.earnings_panels),
        LongExperiment("seasonality", "same-month seasonality", SameMonthSeasonality,
                       {"years": 10, "min_years": 5}, grid),
    ]
    for name, sig, largest in (("value", "book_to_market", True), ("profitability", "gross_profitability", True),
                               ("asset_growth", "asset_growth", False),
                               ("op_profitability", "operating_profitability", True)):
        out.append(LongExperiment(name, sig, FundamentalSort, {"signal": sig, "largest": largest, "every": "M"},
                                  grid, prepare=longrun.fundamental_panels))
    return {e.name: e for e in out}


class RankBand:
    """Equal weight in ranks `lo`..`hi` of a strategy: hi x top-hi weights minus (lo-1) x top-(lo-1)
    weights (the same ranking), negatives dropped, rows rescaled to sum to 1."""

    def __init__(self, make, lo: int, hi: int):
        if not 1 <= lo <= hi:
            raise ValueError("require 1 <= lo <= hi")
        self.make, self.lo, self.hi = make, lo, hi

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        w = self.make(self.hi).weights(closes)
        if self.lo == 1:
            return w
        w = (self.hi * w - (self.lo - 1) * self.make(self.lo - 1).weights(closes)).clip(lower=0)
        s = w.sum(axis=1)
        return w.div(s.where(s > 0, 1.0), axis=0)


class BufferBand:
    """Buy/hold buffer (sergio's idea, as `buffered.py` for momentum, Novy-Marx & Velikov 2016) for any
    strategy: hold `top_n` names; a held name is sold only when it leaves the top `keep_n`, and the free
    places go to the best-ranked names not held. Ranks 1..top_n come from the strategy's own top-k
    portfolios (the name in top k but not top k-1 has rank k); membership of the top `keep_n` likewise."""

    def __init__(self, make, top_n: int, keep_n: int):
        if not 1 <= top_n <= keep_n:
            raise ValueError("require 1 <= top_n <= keep_n")
        self.make, self.top_n, self.keep_n = make, top_n, keep_n

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        top = [self.make(k).weights(closes).reindex(columns=closes.columns).fillna(0).to_numpy() > 0
               for k in range(1, self.top_n + 1)]
        keep = self.make(self.keep_n).weights(closes).reindex(columns=closes.columns).fillna(0).to_numpy() > 0
        rank = np.full(keep.shape, np.inf)
        for k in range(self.top_n, 0, -1):
            rank[top[k - 1]] = k                     # smallest k containing the name
        change = np.ones(len(closes), dtype=bool)
        change[1:] = (keep[1:] != keep[:-1]).any(axis=1) | (top[-1][1:] != top[-1][:-1]).any(axis=1)
        out = np.zeros(keep.shape)
        held = np.zeros(keep.shape[1], dtype=bool)
        for d in range(len(closes)):
            if change[d]:
                held &= keep[d]                      # sell only names out of the top keep_n
                free = self.top_n - int(held.sum())
                if free > 0:
                    cand = np.flatnonzero(~held & np.isfinite(rank[d]))
                    held[cand[np.argsort(rank[d, cand], kind="stable")][:free]] = True
            if held.any():
                out[d, held] = 1.0 / held.sum()
        return pd.DataFrame(out, index=closes.index, columns=closes.columns)


def fingerprint() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    for mod in ("longrun", "portfolio", "costs", "stats", "factors", "combo", "fundamentals", "pead", "seasonal",
                "longrun_combo"):
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    return h.hexdigest()


def run(out: Path, history=longrun.HISTORY):
    import importlib
    for mod in ("combo", "fundamentals", "pead", "seasonal", "longrun_combo"):
        importlib.import_module(f"algotrader.{mod}")
    frames = longrun.load_frames(history)
    members = longrun.membership_matrix(longrun.parse_members((Path(history) / "sp500_members.csv").read_text()))
    out.mkdir(parents=True, exist_ok=True)
    cols, turnover = {}, {}
    for name, exp in strategies().items():
        extra = exp.prepare(frames, members) if exp.prepare else None
        make = lambda k, exp=exp, extra=extra: exp.make({"top_n": k}, extra)
        jobs = [(f"top{k}", RankBand(make, 1, k), COST_VARIANTS) for k in SIZES]
        jobs += [(f"band{lo}-{hi}", RankBand(make, lo, hi), ("base",)) for lo, hi in BANDS]
        jobs += [(f"buf{n}-{k}", BufferBand(make, n, k), COST_VARIANTS) for n, k in BUFFERS]
        for label, strat, costs in jobs:
            for var in costs:
                res = PortfolioBacktester(PointInTime(strat, members), longrun.COSTS[var], longrun.CAPITAL).run(frames)
                cols[f"{name}|{label}|{var}"] = longrun._returns(res.equity, None, None)
                if var == "base":
                    turnover[f"{name}|{label}"] = longrun.turnover(res, TEST_START, None)
            print(f"{name} {label}: Sharpe {returns_stats(cols[f'{name}|{label}|base'][TEST_START:])['sharpe']:.2f}",
                  flush=True)
    spy = frames[longrun.BENCHMARK]["close"].pct_change()
    df = pd.DataFrame(cols)
    df["SPY"] = spy.reindex(df.index)
    df.to_csv(out / "returns.csv", float_format="%.6g")
    n_conf = len(strategies()) * (len(SIZES) + len(BANDS) + len(BUFFERS)) + 6
    (out / "manifest.json").write_text(json.dumps(
        {"concentration": {"code_sha256": fingerprint(), "data_sha256": longrun.data_fingerprint(history),
                           "configurations": n_conf, "turnover": turnover}}, indent=2, sort_keys=True))


# ---------------------------------------------------------------- combinations and report

def yearly_pick(r: pd.DataFrame, k: int, start=TEST_START) -> tuple:
    """Each January from `start`, the `k` columns of `r` with the best Sharpe on all earlier days;
    returns (their equal-weight daily return from that January to the next, {year: picks})."""
    out, picks = [], {}
    years = sorted({d.year for d in r.index if d >= start})
    for y in years:
        F, E = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        past = r[r.index < F]
        s = past.mean() / past.std() * np.sqrt(252)
        best = list(s.dropna().sort_values(ascending=False).index[:k])
        picks[y] = best
        out.append(r.loc[(r.index >= F) & (r.index < E), best].mean(axis=1))
    return pd.concat(out), picks


def combinations(df: pd.DataFrame, var="base") -> tuple:
    names = sorted({c.split("|")[0] for c in df.columns if "|" in c})
    out, picks = {}, {}
    for k in (5, 20):
        r = df[[f"{n}|top{k}|{var}" for n in names]]
        r.columns = names
        out[f"all {len(names)} strategies, top {k} each"] = r.mean(axis=1)
        out[f"best 5 strategies (chosen yearly), top {k} each"], picks[k] = yearly_pick(r, 5)
    r = df[[f"{n}|top5|{var}" for n in names]]
    r.columns = names
    out["best 1 strategy (chosen yearly), top 5"], _ = yearly_pick(r, 1)
    if all(f"{n}|buf5-20|{var}" in df for n in names):
        r = df[[f"{n}|buf5-20|{var}" for n in names]]
        r.columns = names
        out["best 5 strategies (chosen yearly), top 5 kept while in top 20"], _ = yearly_pick(r, 5)
    return out, picks


def _worst_month(r):
    return float((1 + r).groupby(r.index.to_period("M")).prod().min() - 1)


def _row(label, r, spy, factors, stress=None, turnover=None):
    from .french import factor_regression
    r = r[r.index >= TEST_START].dropna()
    st = returns_stats(r)
    lo, hi = sharpe_interval(r)
    ff = factor_regression(r, factors) if factors is not None else {"alpha": np.nan, "alpha_t": np.nan}
    ab = alpha_beta(r, spy)
    cells = [label, f"{st['sharpe']:.2f} [{lo:.2f}, {hi:.2f}]", f"{st['cagr']:+.1%}", f"{r.std() * np.sqrt(252):.1%}",
             f"{st['max_drawdown']:.1%}", f"{_worst_month(r):.1%}", f"{ab['beta']:.2f}",
             f"{ff['alpha']:+.1%} ({ff['alpha_t']:.2f})",
             f"{returns_stats(stress[stress.index >= TEST_START])['sharpe']:.2f}" if stress is not None else "",
             f"{turnover:.0%}" if turnover is not None else ""]
    return "| " + " | ".join(cells) + " |"


HEAD = ("| {first} | Sharpe [90% CI] | CAGR | Volatility | Max DD | Worst month | Beta | FF alpha/yr (t) | "
        "Stress Sharpe | Turnover |\n|---|---|---|---|---|---|---|---|---|---|")


def report(out: Path, data_dir="data/french"):
    from .french import factor_regression, load
    df = pd.read_csv(out / "returns.csv", index_col=0, parse_dates=True)
    man = json.loads((out / "manifest.json").read_text())["concentration"]
    turnover = man["turnover"]
    factors = load(data_dir) if Path(data_dir).exists() else None
    spy = df["SPY"]
    names = sorted({c.split("|")[0] for c in df.columns if "|" in c})
    test = df[df.index >= TEST_START]

    # 1. average over strategies of each size
    avg = ["| N | Mean Sharpe | Median Sharpe | Mean CAGR | Mean volatility | Mean max DD | Mean worst month | "
           "Strategies where it is the best N |", "|---|---|---|---|---|---|---|---|"]
    best_n = {n: max(SIZES, key=lambda k: returns_stats(test[f"{n}|top{k}|base"])["sharpe"]) for n in names}
    for k in SIZES:
        st = [returns_stats(test[f"{n}|top{k}|base"]) for n in names]
        vol = [test[f"{n}|top{k}|base"].std() * np.sqrt(252) for n in names]
        wm = [_worst_month(test[f"{n}|top{k}|base"].dropna()) for n in names]
        avg.append(f"| {k} | {np.mean([s['sharpe'] for s in st]):.2f} | {np.median([s['sharpe'] for s in st]):.2f} | "
                   f"{np.mean([s['cagr'] for s in st]):+.1%} | {np.mean(vol):.1%} | "
                   f"{np.mean([s['max_drawdown'] for s in st]):.1%} | {np.mean(wm):.1%} | "
                   f"{sum(1 for n in names if best_n[n] == k)} |")

    # 2. concentration minus top 50, per strategy (the pre-registered test)
    diff = ["| Strategy | Top 3 - top 50: return/yr (t) | FF alpha/yr (t) | Top 5 - top 50: return/yr (t) | "
            "FF alpha/yr (t) | Passes |", "|---|---|---|---|---|---|"]
    passes = 0
    for n in names:
        cells, ok = [f"`{n}`"], False
        for k in (3, 5):
            d = (test[f"{n}|top{k}|base"] - test[f"{n}|top50|base"]).dropna()
            t = d.mean() / d.std() * np.sqrt(len(d))
            ff = factor_regression(d, factors) if factors is not None else {"alpha": np.nan, "alpha_t": np.nan}
            ok |= ff["alpha_t"] >= 2
            cells += [f"{d.mean() * 252:+.1%} ({t:.2f})", f"{ff['alpha']:+.1%} ({ff['alpha_t']:.2f})"]
        passes += ok
        diff.append("| " + " | ".join(cells + ["yes" if ok else "no"]) + " |")

    # 2b. buffers against the plain top N
    buf = ["| Strategy | " + " | ".join(f"Top {b} -> kept in top {k}: Sharpe, turnover" for b, k in BUFFERS) + " |",
           "|---" * (len(BUFFERS) + 1) + "|"]
    for n in names:
        cells = []
        for b, k in BUFFERS:
            if f"{n}|buf{b}-{k}|base" not in test:
                cells.append("")
                continue
            sp, sb = (returns_stats(test[f"{n}|{lab}|base"])["sharpe"] for lab in (f"top{b}", f"buf{b}-{k}"))
            cells.append(f"{sp:.2f} -> {sb:.2f}, {turnover.get(f'{n}|top{b}', np.nan):.0%} -> "
                         f"{turnover.get(f'{n}|buf{b}-{k}', np.nan):.0%}")
        buf.append(f"| `{n}` | " + " | ".join(cells) + " |")

    # 3. rank bands: annual mean return of each band
    labels = ["top3"] + [f"band{lo}-{hi}" for lo, hi in BANDS]
    band = ["| Strategy | Ranks 1-3 | 4-5 | 6-10 | 11-20 | 21-50 | 1-3 minus 21-50 (t) |", "|---|---|---|---|---|---|---|"]
    for n in names:
        m = [test[f"{n}|{lab}|base"].mean() * 252 for lab in labels]
        d = (test[f"{n}|top3|base"] - test[f"{n}|band21-50|base"]).dropna()
        band.append(f"| `{n}` | " + " | ".join(f"{x:+.1%}" for x in m) +
                    f" | {d.mean() * 252:+.1%} ({d.mean() / d.std() * np.sqrt(len(d)):.2f}) |")

    # 4. per strategy and size
    per = []
    for n in names:
        per.append(f"### `{n}`\n\n" + HEAD.format(first="N"))
        for k in SIZES:
            per.append(_row(f"top {k}", df[f"{n}|top{k}|base"], spy, factors, df[f"{n}|top{k}|stress"],
                            turnover.get(f"{n}|top{k}")))
        for b, k in BUFFERS:
            if f"{n}|buf{b}-{k}|base" in df:
                per.append(_row(f"top {b}, kept while in top {k}", df[f"{n}|buf{b}-{k}|base"], spy, factors,
                                df[f"{n}|buf{b}-{k}|stress"], turnover.get(f"{n}|buf{b}-{k}")))
        per.append("")

    # 5. combinations
    comb, picks = combinations(df)
    comb_s, _ = combinations(df, "stress")
    combo_rows = [HEAD.format(first="Portfolio")]
    for label, r in comb.items():
        combo_rows.append(_row(label, r, spy, factors, comb_s[label]))
    combo_rows.append(_row("SPY (buy and hold)", spy, spy, factors))
    pick_lines = "\n".join(f"- {y}: " + ", ".join(p) for y, p in picks[5].items())

    years = len(test.dropna(how="all")) / 252
    text = f"""# Concentration test (long history, point-in-time S&P 500)

Generated by `python -m algotrader.concentration report`; do not edit by hand. Eleven long-only strategies at the
parameters their walk-forward chose most often, each held at N = 3, 5, 10, 20, 50 stocks (equal weight, monthly, $1M,
IBKR + 2 bp base / 5 bp stress), test period {TEST_START.date()} to {test.index[-1].date()} ({years:.1f} years). FF alpha:
Fama-French five factors plus momentum (total market return as the market factor). {man['configurations']}
configurations (counted in `results/trials.md`).

**Pre-registered rule:** concentration is better for a strategy if top 3 or top 5 minus top 50 has a Fama-French alpha
with t >= 2 at base costs; better in general only if that holds for most strategies. **Result: {passes} of {len(names)}
strategies pass.**

## Average over the {len(names)} strategies

{chr(10).join(avg)}

## Concentrated minus top 50 (daily return difference)

{chr(10).join(diff)}

## Buy/hold buffer (buy the top N, sell only when out of the top K)

Plain top N -> buffered, base costs; turnover one-sided per year.

{chr(10).join(buf)}

## Mean return by rank band (per year, arithmetic)

Does the very top of each ranking earn more than the ranks below it?

{chr(10).join(band)}

## Combining strategies

Each portfolio is the average of its strategies' daily returns (parts rebalanced daily, shared names not netted, each
part's own costs). "Chosen yearly": every January the strategies with the best Sharpe on all earlier days (from 2008).

{chr(10).join(combo_rows)}

Best five strategies at top 5, as chosen each January:

{pick_lines}

## Each strategy

{chr(10).join(per)}
"""
    (out / "summary.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "report"])
    p.add_argument("--out", default="results/long_conc")
    p.add_argument("--history", default=str(longrun.HISTORY))
    p.add_argument("--french", default="data/french")
    a = p.parse_args(argv)
    if a.action == "run":
        run(Path(a.out), a.history)
    else:
        report(Path(a.out), a.french)


if __name__ == "__main__":
    main()
