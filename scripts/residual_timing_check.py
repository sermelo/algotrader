"""Does daily residual reversal survive trading at the next open instead of the close?

Contrarian rule on 5-factor PCA residuals of S&P 500 members (WIKI history): weight
proportional to minus the residual's cumulative sum over the last `L` days, mapped to
stock weights with sum |w| = 1 (as in deep_arb). Gross daily returns measured with two
timings: decided at the close of t and earned close(t) -> close(t+1) (the paper's
convention), or traded at the open of t+1 and held to the open of t+2 (ours).

    nice uv run python scripts/residual_timing_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader import longrun
from algotrader.deep_arb import pca_residuals, stock_weights
from algotrader.history import parse_members

fr = longrun.load_frames("data/history_wiki")
fr.pop(longrun.BENCHMARK, None)
mem = longrun.membership_matrix(parse_members(Path("data/history_wiki/sp500_members.csv").read_text()))
closes = pd.DataFrame({t: f["close"] for t, f in fr.items()})
opens = pd.DataFrame({t: f["open"] for t, f in fr.items()})
rets = closes.pct_change(fill_method=None)
m = mem.reindex(columns=closes.columns, fill_value=False).astype(float)
daily = m.reindex(index=closes.index.union(m.index)).ffill().reindex(closes.index).fillna(0.0) > 0.5
eps, load = pca_residuals(rets, daily, k=5)
E = eps.to_numpy(float)
pos = {d: i for i, d in enumerate(closes.index)}
cc = rets.shift(-1).to_numpy(float)                                   # close t -> close t+1
oo = (opens.shift(-2) / opens.shift(-1) - 1).to_numpy(float)          # open t+1 -> open t+2
for L in (1, 5, 30):
    rows = []
    for d, (cols, W, beta) in load.items():
        t = pos[d]
        if t < L:
            continue
        idx = np.array([closes.columns.get_loc(c) for c in cols])
        win = E[t - L + 1:t + 1, idx]
        ok = np.isfinite(win).all(axis=0)
        a = -np.nan_to_num(win.sum(axis=0)) * ok
        w = stock_weights(a, W, beta)
        rows.append((d, np.nansum(w * cc[t, idx]), np.nansum(w * oo[t, idx])))
    r = pd.DataFrame(rows, columns=["date", "close_to_close", "open_to_open"]).set_index("date")
    r = r.loc["2010":]
    s = r.mean() / r.std() * np.sqrt(252)
    print(f"L={L:2d}  gross Sharpe close-to-close {s['close_to_close']:.2f}  "
          f"open-to-open {s['open_to_open']:.2f}  mean/yr {r.mean()['close_to_close']*252:+.1%} / "
          f"{r.mean()['open_to_open']*252:+.1%}", flush=True)
