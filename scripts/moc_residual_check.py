"""Does daily residual reversal survive a market-on-close order decided at 15:45?

Follow-up to `residual_timing_check.py` (roadmap S8). That check, on WIKI daily bars,
found most of the reversal is earned overnight, so trading at the next open loses it.
The paper's timing (decide on the close, earn close -> close) is only reachable with a
signal computed a few minutes before the close and a market-on-close order. Our
1-minute IEX bars (2022 on) allow that test.

Same rule as the timing check: 5-factor PCA residuals (252-day window), allocation
minus the cumulative residual of the last L days, mapped to stock weights with
sum |w| = 1 (market neutral). Universe: the broad universe without ETFs. Three timings,
same loadings:

- paper:    residuals from close(t), earned close(t) -> close(t+1) (not tradable);
- moc:      day t's residual from the price at 15:45 (the close of the last bar starting
            by 15:44, 12:44 on half days) over close(t-1), earlier days' from closes;
            earned close(t) -> close(t+1) (MOC fill at the closing price);
- next_open: residuals from close(t), earned open(t+1) -> open(t+2) (our daily engine).

Net returns charge `bps` per unit of one-sided turnover (sum |w_t - w_t-1|, ignoring
drift) and 50 bp a year on the short half. Gross Sharpe ignores both.

    nice uv run python scripts/moc_residual_check.py
"""
from datetime import time

import numpy as np
import pandas as pd

from algotrader import BarLoader
from algotrader.deep_arb import pca_residuals, stock_weights
from algotrader.universe import BROAD

ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY",
        "XLU", "XLB", "GLD", "TLT", "ITB", "SDS", "SMH", "TIP", "XBI"}
SNAP_BEFORE_CLOSE = pd.Timedelta(minutes=15)    # signal price: bar ending 15 min before the close
BORROW = 0.005


def daily_prices(bars: pd.DataFrame, session_close: pd.Series) -> pd.DataFrame:
    """Per session: first open, last close and the snapshot price 15 minutes before the
    session close (close of the last bar ending by then; NaN if none)."""
    day = bars.index.normalize()
    end = bars.index + pd.Timedelta(minutes=1)                  # bar end times
    cutoff = pd.DatetimeIndex(session_close.reindex(day).to_numpy()) - SNAP_BEFORE_CLOSE
    g = bars.groupby(day)
    out = pd.DataFrame({"open": g["open"].first(), "close": g["close"].last()})
    snap = bars["close"][end <= cutoff].groupby(day[end <= cutoff]).last()
    out["snap"] = snap.reindex(out.index)
    return out


def session_closes(spy: pd.DataFrame) -> pd.Series:
    """Session close time per day (16:00, or 13:00 on half days) from SPY's last bar."""
    last = spy.index.to_series().groupby(spy.index.normalize()).max()
    close = last + pd.Timedelta(minutes=1)
    # a session whose last SPY bar ends before 14:00 is a half day (13:00 close)
    half = close.dt.time < time(14, 0)
    return close.where(half, close.dt.normalize() + pd.Timedelta(hours=16))


def main():
    loader = BarLoader("data/scratch/tickers")
    sc = session_closes(loader.load("SPY", timeframe="1min"))
    frames = {}
    for t in BROAD:
        if t in ETFS:
            continue
        try:
            frames[t] = daily_prices(loader.load(t, timeframe="1min"), sc)
        except FileNotFoundError:
            print("missing", t)
    closes = pd.DataFrame({t: f["close"] for t, f in frames.items()}).sort_index()
    opens = pd.DataFrame({t: f["open"] for t, f in frames.items()}).reindex(closes.index)
    snaps = pd.DataFrame({t: f["snap"] for t, f in frames.items()}).reindex(closes.index)
    print(f"{closes.shape[1]} stocks, {len(closes)} sessions "
          f"({closes.index[0].date()} to {closes.index[-1].date()})", flush=True)

    rets = closes.pct_change(fill_method=None)
    snap_ret = (snaps / closes.shift(1) - 1).to_numpy(float)
    eps, load = pca_residuals(rets, k=5)
    E = eps.to_numpy(float)
    pos = {d: i for i, d in enumerate(closes.index)}
    cc = rets.shift(-1).to_numpy(float)                                  # close t -> t+1
    oo = (opens.shift(-2) / opens.shift(-1) - 1).to_numpy(float)         # open t+1 -> t+2

    for L in (1, 5, 30):
        rows, prev = [], {"paper": None, "moc": None, "next_open": None}
        for d, (cols, W, beta) in load.items():
            t = pos[d]
            if t < L:
                continue
            idx = np.array([closes.columns.get_loc(c) for c in cols])
            win = E[t - L + 1:t + 1, idx]
            r_snap = snap_ret[t, idx]
            e_snap = np.where(np.isfinite(r_snap), r_snap, 0.0)
            e_snap = e_snap - beta @ (W.T @ e_snap)                       # residual at 15:45
            win_moc = win.copy()
            win_moc[-1] = np.where(np.isfinite(r_snap), e_snap, np.nan)
            row = {"date": d}
            for name, w_in, fwd in (("paper", win, cc), ("moc", win_moc, cc),
                                    ("next_open", win, oo)):
                ok = np.isfinite(w_in).all(axis=0)
                a = -np.nan_to_num(w_in.sum(axis=0)) * ok
                w = pd.Series(stock_weights(a, W, beta), index=cols)
                p = prev[name]
                turn = w.abs().sum() if p is None else w.sub(p, fill_value=0).abs().sum()
                prev[name] = w
                row[name] = np.nansum(w.to_numpy() * fwd[t, idx])
                row[name + "_turn"] = turn
                row[name + "_short"] = -w[w < 0].sum()
            rows.append(row)
        r = pd.DataFrame(rows).set_index("date").iloc[1:]
        print(f"\nL={L}  ({r.index[0].date()} to {r.index[-1].date()}, {len(r)} days)")
        for name in ("paper", "moc", "next_open"):
            g = r[name]
            turn = r[name + "_turn"]
            line = (f"  {name:9s} gross Sharpe {g.mean() / g.std() * np.sqrt(252):5.2f}"
                    f"  mean/yr {g.mean() * 252:+6.1%}  turnover/yr {turn.mean() * 252:5.0f}x")
            for bps in (2, 5):
                n = g - turn * bps / 1e4 - r[name + "_short"] * BORROW / 252
                line += f"  net@{bps}bp {n.mean() / n.std() * np.sqrt(252):5.2f}"
            yearly = g.groupby(g.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252))
            line += "  by year " + " ".join(f"{y}:{v:.1f}" for y, v in yearly.items())
            print(line, flush=True)


if __name__ == "__main__":
    main()
