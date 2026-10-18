"""Blends of the long-history strategies' walk-forward daily returns (base costs).

Pre-registered blends (no search over weights): equal weight of all long-only strategies; of the
long-short strategies (market neutral); of long-only + long-short; and inverse-volatility weights
(trailing 252 days, known the day before) of the same. Reported next to SPY and the best single
strategy. Each member already pays its own costs, so a blend is slightly conservative (trades that
would net against each other are not netted).

    python scripts/strategy_blend.py results/long results/long/blend.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.stats import alpha_beta, returns_stats

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
EXCLUDE = ("deep_arb_long",)                       # lost 97% by itself; a blend would only show that
R, SPY = {}, None
for f in sorted(RES.glob("*_wf_returns.csv")):
    name = f.name.replace("_wf_returns.csv", "")
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    if name in EXCLUDE:
        continue
    R[name] = d["base"]
    SPY = d["hold"] if SPY is None else SPY.combine_first(d["hold"])
R = pd.DataFrame(R)
common = R.dropna().index                          # days every strategy has a return (chart CNN starts 2014)
R, SPY = R.loc[common], SPY.reindex(common)
longs = [c for c in R if "_ls_" not in c and c not in ("chart_cnn_ls_long",)]
shorts = [c for c in R if c not in longs]


def inv_vol(df):
    vol = df.rolling(252, min_periods=60).std().shift(1)
    w = (1 / vol).replace([np.inf, -np.inf], np.nan)
    w = w.div(w.sum(axis=1), axis=0)
    return (df * w).sum(axis=1, min_count=1).dropna()


blends = {
    "equal: all long-only": R[longs].mean(axis=1),
    "equal: all long-short": R[shorts].mean(axis=1),
    "equal: long-only + long-short": R.mean(axis=1),
    "inverse-vol: long-only": inv_vol(R[longs]),
    "inverse-vol: long-short": inv_vol(R[shorts]),
    "inverse-vol: all": inv_vol(R),
    "long-only 70% + long-short 30%": 0.7 * R[longs].mean(axis=1) + 0.3 * R[shorts].mean(axis=1),
}
rows = []
def add(name, r):
    r = r.dropna()
    s = returns_stats(r)
    a = alpha_beta(r, SPY.reindex(r.index))
    rows.append((name, s["sharpe"], s["cagr"], s["max_drawdown"], a["alpha"], a["alpha_t"], a["beta"]))
add("SPY buy and hold", SPY)
best = R.apply(lambda c: returns_stats(c)["sharpe"]).idxmax()
add(f"best single ({best})", R[best])
for k, v in blends.items():
    add(k, v)
corr = R.corr()
lines = ["# Blends of the long-history strategies", "",
         f"Daily walk-forward returns (base costs), {common[0].date()} to {common[-1].date()}, {len(common)} days, "
         f"{R.shape[1]} strategies ({len(longs)} long-only, {len(shorts)} long-short; `deep_arb_long` left out).", "",
         "| Portfolio | Sharpe | CAGR | Max DD | Alpha/yr vs SPY (t) | Beta |", "|---|---|---|---|---|---|"]
for n, sh, c, dd, a, t, b in rows:
    al = "–" if n.startswith("SPY") else f"{a:+.1%} ({t:.2f})"
    lines.append(f"| {n} | {sh:.2f} | {c:+.1%} | {dd:.1%} | {al} | {b:.2f} |")
off = corr.where(~np.eye(len(corr), dtype=bool))
lines += ["", f"Average pairwise correlation: long-only {off.loc[longs, longs].stack().mean():.2f}, "
          f"long-short {off.loc[shorts, shorts].stack().mean():.2f}, long-only with long-short "
          f"{off.loc[longs, shorts].stack().mean():.2f}."]
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
