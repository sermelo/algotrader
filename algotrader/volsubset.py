"""Volatile subset: the usual stock pickers, but only among the most volatile S&P 500 members.

Question (sergio, 2026-10-22): the strategies need prices that move a lot, so would they do better if
they picked only from the volatile stocks, checked more often (weekly or daily) because those prices
change faster? On the long point-in-time S&P 500 history, two strategies (12-1 month momentum, the one
the literature finds strongest among volatile stocks (Zhang 2006, "information uncertainty"), and
operating profitability, the best long-history strategy here) at top 5 and top 20, rebalanced at month,
week or day ends, each picking from (a) all members and (b) only the `FRAC` most volatile members of
that day. Volatility = standard deviation of the last `VOL_LOOKBACK` daily close-to-close returns, known
at that day's close (at least `VOL_MIN` returns); a stock that leaves the volatile subset is sold at the
next rebalance. Same engine, $1M, IBKR commission + 0 / 2 / 5 bp half-spread (gross / base / stress).

Pre-registered (2026-10-22, before any run), on the test period 2012-01-01 on, base costs unless said:
1. **Edge:** a volatile-subset configuration has an edge if its Fama-French five-factor + momentum alpha
   has t >= 2 at base costs and is still positive at stress costs.
2. **The filter helps** a strategy if volatile minus all-members (same size and frequency, daily returns)
   has a Fama-French alpha with t >= 2 for most of its six size/frequency pairs.
3. **Checking more often helps** within the volatile subset if weekly or daily minus monthly (same size)
   has a Fama-French alpha with t >= 2 for most of the four strategy/size pairs.
12 new volatile-subset configurations plus 4 all-member daily ones (the monthly and weekly all-member
ones repeat the weekly test); one-sided, about 2.3% of zero-edge tests reach t >= 2 by chance.

    python -m algotrader.volsubset run --out data/long_vol         # backtests (VPS) -> returns.csv
    python -m algotrader.volsubset report --out results/long_vol   # tables -> summary.md
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import concentration, longrun
from .concentration import HEAD, TEST_START, _row
from .longrun import PointInTime
from .portfolio import PortfolioBacktester, _top_weights
from .stats import returns_stats

FRAC = 1 / 3
VOL_LOOKBACK = 63
VOL_MIN = 40
NAMES = ("momentum", "op_profitability")
FREQS = ("M", "W", "D")
SIZES = (5, 20)
UNIVERSES = ("all", "vol")
COST_VARIANTS = ("gross", "base", "stress")
DEPS = ("fundamentals",)


def volatile(closes: pd.DataFrame, frac=FRAC, lookback=VOL_LOOKBACK, min_periods=VOL_MIN) -> pd.DataFrame:
    """Bool, dates x tickers: in the `frac` most volatile of the tickers trading that day (the
    standard deviation of the last `lookback` daily returns, from closes up to that day)."""
    vol = closes.pct_change(fill_method=None).rolling(lookback, min_periods=min_periods).std()
    vol = vol.where(closes.notna())
    return vol.rank(axis=1, pct=True, ascending=False) <= frac


class VolatileSubset:
    """A top-N strategy that may only pick from the volatile tickers: its own score, hidden outside
    `volatile(closes)`, ranked with the same rule (`_top_weights`) at its own rebalance days.
    The inner strategy needs `top_n`, `every` and either `score(closes)` or `panels`/`signal`."""

    def __init__(self, inner, frac=FRAC, lookback=VOL_LOOKBACK, min_periods=VOL_MIN):
        if not 0 < frac <= 1 or lookback < 2 or not 2 <= min_periods <= lookback:
            raise ValueError("require 0 < frac <= 1, lookback >= 2, 2 <= min_periods <= lookback")
        self.inner, self.frac, self.lookback, self.min_periods = inner, frac, lookback, min_periods

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.inner, "score"):
            return self.inner.score(closes)
        return self.inner.panels[self.inner.signal].reindex(index=closes.index, columns=closes.columns)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        keep = volatile(closes, self.frac, self.lookback, self.min_periods)
        return _top_weights(self.score(closes).where(keep), closes, self.inner.top_n, self.inner.every,
                            largest=getattr(self.inner, "largest", True))


def strategies() -> dict:
    return {n: e for n, e in concentration.strategies().items() if n in NAMES}


def make(exp, extra, every, k, universe):
    strat = exp.strategy(**dict(exp.fixed, every=every), **(extra or {}), top_n=k)
    return VolatileSubset(strat) if universe == "vol" else strat


def n_configurations() -> int:
    """New tries: every volatile-subset configuration and the all-member daily ones."""
    return len(NAMES) * len(SIZES) * (len(FREQS) + 1)


def fingerprint() -> str:
    for mod in DEPS:
        importlib.import_module(f"algotrader.{mod}")
    h = hashlib.sha256(Path(__file__).read_bytes())
    for mod in ("concentration", "longrun", "portfolio", "costs", "stats", "factors") + DEPS:
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    return h.hexdigest()


def run(out: Path, history=longrun.HISTORY):
    fp = fingerprint()
    frames = longrun.load_frames(history)
    members = longrun.membership_matrix(longrun.parse_members((Path(history) / "sp500_members.csv").read_text()))
    out.mkdir(parents=True, exist_ok=True)
    cols, turnover = {}, {}
    for name, exp in strategies().items():
        extra = exp.prepare(frames, members) if exp.prepare else None
        for universe in UNIVERSES:
            for every in FREQS:
                for k in SIZES:
                    key = f"{name}|{universe}|top{k}|{every}"
                    strat = make(exp, extra, every, k, universe)
                    for var in COST_VARIANTS:
                        res = PortfolioBacktester(PointInTime(strat, members), longrun.COSTS[var],
                                                  longrun.CAPITAL).run(frames)
                        cols[f"{key}|{var}"] = longrun._returns(res.equity, None, None)
                        if var == "base":
                            turnover[key] = longrun.turnover(res, TEST_START, None)
                    print(f"{key}: Sharpe {_sharpe(cols[f'{key}|base']):.2f}", flush=True)
    df = pd.DataFrame(cols)
    df["SPY"] = frames[longrun.BENCHMARK]["close"].pct_change().reindex(df.index)
    df.to_csv(out / "returns.csv", float_format="%.6g")
    (out / "manifest.json").write_text(json.dumps(
        {"volsubset": {"code_sha256": fp, "data_sha256": longrun.data_fingerprint(history),
                       "configurations": n_configurations(), "turnover": turnover}}, indent=2, sort_keys=True))


# ---------------------------------------------------------------- report

def _test(r):
    return r[r.index >= TEST_START].dropna()


def _sharpe(r):
    return returns_stats(_test(r))["sharpe"]


def _ff(r, factors):
    from .french import factor_regression
    if factors is None:
        return {"alpha": np.nan, "alpha_t": np.nan}
    return factor_regression(_test(r), factors)


def _diff_row(label, a, b, factors):
    d = _test(a - b)
    ff = _ff(d, factors)
    t = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan
    return {"label": label, "diff": d.mean() * 252, "diff_t": t, "ff_alpha": ff["alpha"], "ff_t": ff["alpha_t"]}


def verdicts(df: pd.DataFrame, factors=None) -> dict:
    """The three pre-registered tests: {'edge': rows, 'filter': rows, 'freq': rows}, each row with a
    `pass` flag, plus the counts."""
    names = sorted({c.split("|")[0] for c in df.columns if "|" in c})
    edge, filt, freq = [], [], []
    for n in names:
        for k in SIZES:
            for f in FREQS:
                key = f"{n}|vol|top{k}|{f}"
                if f"{key}|base" not in df:
                    continue
                b, s = _ff(df[f"{key}|base"], factors), _ff(df[f"{key}|stress"], factors)
                edge.append({"strategy": n, "label": f"top{k} {f}", "ff_alpha": b["alpha"], "ff_t": b["alpha_t"],
                             "stress_alpha": s["alpha"],
                             "pass": bool(b["alpha_t"] >= 2 and s["alpha"] > 0)})
                row = _diff_row(f"top{k} {f}", df[f"{key}|base"], df[f"{n}|all|top{k}|{f}|base"], factors)
                filt.append({"strategy": n, **row, "pass": bool(row["ff_t"] >= 2)})
            for f in FREQS[1:]:
                row = _diff_row(f"top{k} {f} - M", df[f"{n}|vol|top{k}|{f}|base"],
                                df[f"{n}|vol|top{k}|M|base"], factors)
                freq.append({"strategy": n, "freq": f, **row, "pass": bool(row["ff_t"] >= 2)})
    filter_pass = [n for n in names if sum(r["pass"] for r in filt if r["strategy"] == n)
                   > len([r for r in filt if r["strategy"] == n]) / 2]
    freq_pass = [f for f in FREQS[1:] if sum(r["pass"] for r in freq if r["freq"] == f)
                 > len([r for r in freq if r["freq"] == f]) / 2]
    return {"edge": edge, "filter": filt, "freq": freq, "names": names,
            "edge_pass": [f"{r['strategy']} {r['label']}" for r in edge if r["pass"]],
            "filter_pass": filter_pass, "freq_pass": freq_pass}


def report(out: Path, data_dir="data/french") -> str:
    from .french import load
    df = pd.read_csv(out / "returns.csv", index_col=0, parse_dates=True)
    man = json.loads((out / "manifest.json").read_text())["volsubset"]
    to = man["turnover"]
    factors = load(data_dir) if Path(data_dir).exists() else None
    v = verdicts(df, factors)
    spy = df["SPY"]
    end = df.index.max().date()
    years = (df.index.max() - TEST_START).days / 365.25
    fname = {"M": "monthly", "W": "weekly", "D": "daily"}

    per = []
    for n in v["names"]:
        per += [f"### `{n}`", "", HEAD.format(first="Version")]
        for k in SIZES:
            for u in UNIVERSES:
                for f in FREQS:
                    c = f"{n}|{u}|top{k}|{f}"
                    if f"{c}|base" in df:
                        lab = f"top {k} {fname[f]}, {'volatile third' if u == 'vol' else 'all members'}"
                        per.append(_row(lab, df[f"{c}|base"], spy, factors, df[f"{c}|stress"], to[c]))
        per.append(_row("SPY (buy and hold)", spy, spy, factors))
        per.append("")

    def table(rows, first):
        lines = [f"| Strategy | {first} | Difference/yr (t) | FF alpha/yr (t) | Pass |", "|---|---|---|---|---|"]
        lines += [f"| `{r['strategy']}` | {r['label']} | {r['diff']:+.1%} ({r['diff_t']:.2f}) | "
                  f"{r['ff_alpha']:+.1%} ({r['ff_t']:.2f}) | {'yes' if r['pass'] else 'no'} |" for r in rows]
        return "\n".join(lines)

    edge = ["| Strategy | Version | FF alpha/yr (t), base | FF alpha/yr, stress | Pass |", "|---|---|---|---|---|"]
    edge += [f"| `{r['strategy']}` | {r['label']} | {r['ff_alpha']:+.1%} ({r['ff_t']:.2f}) | "
             f"{r['stress_alpha']:+.1%} | {'yes' if r['pass'] else 'no'} |" for r in v["edge"]]

    text = f"""# Volatile subset (long history, point-in-time S&P 500)

Generated by `python -m algotrader.volsubset report`; do not edit by hand. Momentum 12-1 and operating
profitability at top 5 and top 20, rebalanced monthly, weekly or daily, picking from all S&P 500 members
or only from the most volatile third of them that day ({VOL_LOOKBACK}-day standard deviation of daily
returns, known at the close), $1M, IBKR commission + 0 / 2 / 5 bp half-spread, test period
{TEST_START.date()} to {end} ({years:.1f} years). Turnover one-sided per year. {man['configurations']} new
configurations (counted in `results/trials.md`).

**Pre-registered rules and results:**
1. Edge (volatile-subset Fama-French alpha t >= 2 at base costs, positive at stress):
   **{len(v['edge_pass'])} of {len(v['edge'])} pass ({', '.join(v['edge_pass']) or 'none'}).**
2. The filter helps a strategy (volatile minus all members, FF alpha t >= 2 for most size/frequency pairs):
   **{', '.join(v['filter_pass']) or 'no strategy'}.**
3. Checking more often helps in the volatile subset (weekly or daily minus monthly, FF alpha t >= 2 for
   most strategy/size pairs): **{', '.join(fname[f] for f in v['freq_pass']) or 'neither weekly nor daily'}.**

About 2.3% of zero-edge tests reach t >= 2 by chance, so 1 in {len(v['edge'])} is what luck gives.

## 1. Edge of the volatile-subset versions

{chr(10).join(edge)}

## 2. Volatile third minus all members

{table(v['filter'], 'Version')}

## 3. More frequent minus monthly, volatile third

{table(v['freq'], 'Version')}

## Each strategy

{chr(10).join(per)}
"""
    (out / "summary.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("run", "report"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--history", default=str(longrun.HISTORY))
    p.add_argument("--french", default="data/french")
    a = p.parse_args(argv)
    if a.command == "run":
        run(a.out or Path("data/long_vol"), a.history)
    else:
        print(report(a.out or Path("results/long_vol"), a.french))


if __name__ == "__main__":
    main()
