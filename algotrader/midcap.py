"""Mid-cap test of the price-based factor strategies (roadmap: mid-cap test, 2026-10-18).

Universe: `sizeuniverse.py` (each month the 1,500 stocks with the highest dollar volume, minus S&P 500
members and SEC fund tickers; `data/history/size_members.csv`). Prices: the adjusted Hugging Face daily
series of those tickers (`hfdata.build_source` into `data/history_size/hf`): WIKI split and dividend
events up to March 2018 where WIKI has the ticker, splits found from prices after that and for the rest,
NO dividends there (returns understated by about the dividend yield; the equal-weight universe benchmark
has the same bias, SPY does not).

Most of these tickers cannot be checked against a second source, so a ticker reused by a new company
would splice two companies. `segments` cuts a series where trading stops for more than `max_gap`
calendar days, or for more than `jump_gap` days with a price change of more than `max_jump` times; each
piece becomes its own name (`T`, `T~2`, ...) with the parent's membership (a piece has no prices outside
its own dates, so it can only be held while it trades).

Costs are higher than for the S&P 500 runs: half-spread 5 bp (base), 10 bp (stress), none (gross).
Pre-registered (8 experiments, the long-history grids): momentum long and long-short, short-term
reversal, low volatility, 52-week high, seasonality long and long-short, combined price signal.
Success needs a Fama-French alpha with t >= 2 at base costs, a Sharpe above the equal-weight universe
and the result still positive at stress costs; otherwise the factor search stops.

    python -m algotrader.midcap [names] --out data/long_mid
"""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import longrun, sizeuniverse  # noqa: F401 (sizeuniverse is part of the fingerprint)
from .costs import CostModel
from .history import parse_members, wiki_symbol
from .stats import alpha_beta, expected_max_sharpe, returns_stats, sharpe_interval

MEMBERS = Path("data/history/size_members.csv")
PRICES = Path("data/history_size/hf")
SPY_FILE = Path("data/history/tiingo/SPY.parquet")
COSTS = {"base": CostModel(spread_bps=5.0), "gross": CostModel(0.0, 0.0, 0.0, 0.0),
         "stress": CostModel(spread_bps=10.0)}
BASE = ("xs_momentum_long", "xs_momentum_ls_long", "xs_reversal_long", "low_vol_long", "high52_long",
        "seasonality_long", "seasonality_ls_long", "combo_price_long")


def segments(df: pd.DataFrame, max_gap: int = 30, jump_gap: int = 5, max_jump: float = 2.0) -> list:
    """Split a daily frame where the ticker probably changed hands (see module doc)."""
    if len(df) < 2:
        return [df]
    gap = df.index.to_series().diff().dt.days
    ratio = (df["open"] / df["close"].shift()).abs()
    jump = (ratio > max_jump) | (ratio < 1 / max_jump)
    cut = (gap > max_gap) | ((gap > jump_gap) & jump)
    starts = [0] + list(np.flatnonzero(cut.to_numpy()))
    ends = starts[1:] + [len(df)]
    return [df.iloc[a:b] for a, b in zip(starts, ends)]


def load(members_file=MEMBERS, prices=PRICES, spy=SPY_FILE, start=longrun.START, min_days=60):
    """Frames per piece (plus SPY) and the long membership table with the pieces' names."""
    members = parse_members(Path(members_file).read_text())
    frames, names = {}, {}
    for t in sorted(members["ticker"].unique()):
        p = Path(prices) / f"{wiki_symbol(t)}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[[c for c in ("open", "high", "low", "close", "volume", "raw_close") if c in df]]
        df = df[df.index >= pd.Timestamp(start)].dropna(subset=["open", "close"])
        pieces = [s for s in segments(df) if len(s) >= min_days]
        names[t] = []
        for i, s in enumerate(pieces):
            n = t if i == 0 else f"{t}~{i + 1}"
            frames[n] = s
            names[t].append(n)
    rows = [(d, n) for d, t in zip(members["date"], members["ticker"]) for n in names.get(t, [])]
    if Path(spy).exists():
        frames[longrun.BENCHMARK] = pd.read_parquet(spy)[["open", "high", "low", "close", "volume"]].loc[start:]
    return frames, pd.DataFrame(rows, columns=["date", "ticker"])


def equal_weight(frames: dict, member_matrix: pd.DataFrame) -> pd.Series:
    """Daily return of the equal-weight universe (members on the previous day, close to close)."""
    closes = pd.DataFrame({t: f["close"] for t, f in frames.items() if t != longrun.BENCHMARK})
    r = closes.pct_change(fill_method=None)
    m = member_matrix.reindex(columns=closes.columns, fill_value=False).astype(float)
    m = m.reindex(index=closes.index.union(m.index)).ffill().reindex(closes.index).fillna(0.0).shift()
    return r.where(m > 0.5).mean(axis=1)


EXPERIMENTS = []


def experiments():
    if not EXPERIMENTS:
        from . import longrun_combo
        by = dict(longrun.BY_NAME) | {e.name: e for e in longrun_combo.EXPERIMENTS}
        EXPERIMENTS.extend(replace(by[n], name=n.replace("_long", "_mid")) for n in BASE)
    return EXPERIMENTS


def fingerprint(exp) -> str:
    h = hashlib.sha256(longrun.fingerprint(exp).encode())
    h.update(Path(__file__).read_bytes())
    for mod in ("combo", "factors", "seasonal", "sizeuniverse"):
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    return h.hexdigest()


def write_summary(out: Path, ew: pd.Series):
    man = json.loads((out / "manifest.json").read_text())
    n_conf = sum(v["configurations"] for v in man.values())
    rows, years = [], 0.0
    for name in sorted(man):
        wf = pd.read_csv(out / f"{name}_wf_returns.csv", index_col=0, parse_dates=True)
        r = wf["base"].dropna()
        years = max(years, len(r) / 252)
        st, (lo, hi) = returns_stats(r), sharpe_interval(r)
        a_spy, a_ew = alpha_beta(r, wf["hold"]), alpha_beta(r, ew.reindex(r.index))
        folds = pd.read_csv(out / f"{name}_folds.csv")
        rows.append(f"| `{name}` | {st['sharpe']:.2f} [{lo:.2f}, {hi:.2f}] | {st['cagr']:+.1%} | {st['max_drawdown']:.1%} | "
                    f"{a_spy['alpha']:+.1%} ({a_spy['alpha_t']:.2f}) | {a_ew['alpha']:+.1%} ({a_ew['alpha_t']:.2f}) | "
                    f"{a_ew['beta']:.2f} | {returns_stats(wf['gross'].dropna())['sharpe']:.2f} | "
                    f"{returns_stats(wf['stress'].dropna())['sharpe']:.2f} | {folds['turnover'].mean():.0%} | "
                    f"{(folds['sharpe'] > 0).sum()}/{len(folds)} |")
    any_wf = pd.read_csv(out / f"{sorted(man)[0]}_wf_returns.csv", index_col=0, parse_dates=True)
    spy, ewr = returns_stats(any_wf["hold"].dropna()), returns_stats(ew.reindex(any_wf.index).dropna())
    text = f"""# Mid-cap results (ex-S&P 500, dollar-volume universe)

Generated by `python -m algotrader.midcap`; do not edit by hand. Universe: each month the 1,500 stocks with
the highest dollar volume minus S&P 500 members and funds (about 1,040 names); Hugging Face prices with no
dividends after March 2018 (none at all for tickers WIKI lacks). Walk-forward as in `results/long/`; $1,000,000;
costs IBKR + 5 bp half-spread (base), none (gross), 10 bp (stress). Benchmarks over the same days: SPY Sharpe
{spy['sharpe']:.2f} (CAGR {spy['cagr']:+.1%}, with dividends), equal-weight universe Sharpe {ewr['sharpe']:.2f}
(CAGR {ewr['cagr']:+.1%}, same price data). {n_conf} configurations here; the best zero-edge one would show a
Sharpe of about {expected_max_sharpe(n_conf, years):.2f} by luck over {years:.1f} years.

| Experiment | Sharpe [90% CI] | CAGR | Max DD | Alpha/yr vs SPY (t) | Alpha/yr vs EW universe (t) | Beta (EW) | Gross Sharpe | Stress Sharpe | Turnover | Folds > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + "\n"
    (out / "summary.md").write_text(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*")
    p.add_argument("--out", default="data/long_mid")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    exps = {e.name: e for e in experiments()}
    if a.list:
        for e in exps.values():
            print(f"{e.name:24s} {len(e.combos()):3d} configs")
        return
    frames, members_long = load()
    members = longrun.membership_matrix(members_long)
    print(f"{len(frames)} series, {members_long['ticker'].nunique()} names in the lists", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ew = equal_weight(frames, members)
    ew.rename("ew").to_csv(out / "equal_weight_universe.csv", float_format="%.6g")
    saved = longrun.COSTS
    longrun.COSTS = COSTS                              # run_long reads the module's cost table
    try:
        for n in a.names or list(exps):
            res = longrun.run_long(exps[n], frames, members)
            longrun.write(res, out, "data/history_size")
            man = json.loads((out / "manifest.json").read_text())
            man[n]["code_sha256"] = fingerprint(exps[n])
            (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
            print(f"{n}: walk-forward Sharpe {returns_stats(res['wf']['base'])['sharpe']:.2f}", flush=True)
    finally:
        longrun.COSTS = saved
    write_summary(out, ew)


if __name__ == "__main__":
    main()
