"""Stop-loss overlay for the monthly portfolio strategies.

`apply_stop(w, closes, stop, trail, every)` takes a strategy's target weights and sells a
long holding once its close falls `stop` (a fraction, e.g. 0.15) below its reference price:
the close on the day it was bought (decision day), or with `trail` the highest close since
then. The stop is checked on each close and traded at the next open, like every other
signal. A stopped stock stays out (its money in cash) until the next rebalance day, where the
strategy's own choice applies again (a stock still in its top N is bought back).

When a stop fires, the other holdings get their drifted weights (value since the rebalance /
portfolio value), so the engine does not trade them back to equal weight; only rounding and
the overnight gap leave small trades in them. `with_stop(cls)` makes a strategy class with
`stop` (0 = off) and `trail` parameters, so the stop can sit in an experiment's grid.

`stop_events` lists every stop with what holding on until the end of that holding period
would have earned (the gain given up when positive, the loss avoided when negative).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import _rebalance_days


def _periods(w: pd.DataFrame, every) -> np.ndarray:
    """True on days a new holding period starts: rebalance days of `every`, else days the
    weights change."""
    if every:
        out = _rebalance_days(w.index, every)
        out[0] = True                         # weights held from the first day start a period too
        return out
    a = w.to_numpy()
    out = np.zeros(len(w), dtype=bool)
    out[0] = True
    out[1:] = np.abs(np.diff(a, axis=0)).max(axis=1) > 1e-12
    return out


def _run(w: pd.DataFrame, closes: pd.DataFrame, stop: float, trail: bool, every):
    W = w.reindex(index=closes.index, columns=closes.columns).fillna(0.0).to_numpy(float)
    C = closes.ffill().to_numpy(float)
    new = _periods(pd.DataFrame(W, index=closes.index), every)
    out = np.zeros_like(W)
    events = []
    n = W.shape[1]
    ref = np.full(n, np.nan)                  # entry close (or highest close) of each holding
    units = np.zeros(n)                       # holdings in "shares" of a 1.0 portfolio
    cash, row = 1.0, np.zeros(n)
    for d in range(len(W)):
        c = C[d]
        if new[d]:
            target = W[d]
            held = target > 0
            was = units > 0                                  # held and not stopped out
            ref = np.where(held & ~was, c, ref)              # (re)bought: reference = today's close
            ref[~held] = np.nan
            value = cash + np.nansum(units * c)
            ok = np.isfinite(c) & (c > 0)
            units = np.where(held & ok, target * value / np.where(ok, c, 1.0), 0.0)
            cash = value - np.nansum(units * c)
            row = target.copy()
            out[d] = row
            continue
        live = units > 0
        if trail:
            ref = np.where(live & (c > ref), c, ref)
        hit = live & np.isfinite(c) & (c <= ref * (1 - stop)) if stop > 0 else np.zeros(n, dtype=bool)
        if hit.any():
            value = cash + np.nansum(units * c)
            for j in np.flatnonzero(hit):
                events.append((d, j, ref[j], units[j] * c[j] / value))
            cash += float(np.sum(units[hit] * c[hit]))
            units[hit] = 0.0
            value = cash + np.nansum(units * c)
            row = np.where(units > 0, units * np.nan_to_num(c) / value, 0.0)
        out[d] = row
    return out, events, new


def apply_stop(w: pd.DataFrame, closes: pd.DataFrame, stop: float, trail: bool = False,
               every=None) -> pd.DataFrame:
    """Weights `w` with long holdings sold after a close `stop` below their reference."""
    if stop <= 0:
        return w
    out, _, _ = _run(w, closes, stop, trail, every)
    return pd.DataFrame(out, index=closes.index, columns=closes.columns)


def stop_events(w: pd.DataFrame, closes: pd.DataFrame, stop: float, trail: bool = False,
                every=None) -> pd.DataFrame:
    """One row per stop: date, ticker, its weight in the portfolio just before, the close
    that triggered it, its drop from the reference close, `after` = the return from that close
    to the close of the next rebalance day (positive: the gain the stop gave up; negative: the
    loss it avoided), the return since the period started and the trading days left."""
    _, events, new = _run(w, closes, stop, trail, every)
    C = closes.ffill().to_numpy(float)
    starts = np.flatnonzero(new)
    rows = []
    for d, j, ref, weight in events:
        later = starts[starts > d]
        end = later[0] if len(later) else len(C) - 1      # the next rebalance day's close
        entry = C[starts[starts <= d][-1], j] if len(starts[starts <= d]) else np.nan
        rows.append({"date": closes.index[d], "ticker": closes.columns[j], "weight": weight, "close": C[d, j],
                     "drop": C[d, j] / ref - 1, "after": C[end, j] / C[d, j] - 1,
                     "since_period_start": C[d, j] / entry - 1 if entry == entry else np.nan,
                     "days_left": int(end - d)})
    return pd.DataFrame(rows, columns=["date", "ticker", "weight", "close", "drop", "after",
                                       "since_period_start", "days_left"])


def with_stop(cls):
    """Strategy class `cls` with two more parameters: `stop` (fraction below the reference
    close that sells a holding, 0 = off) and `trail` (reference = highest close since entry)."""

    @dataclass(frozen=True)
    class Stopped(cls):
        stop: float = 0.0
        trail: bool = False

        def __post_init__(self):
            if hasattr(super(), "__post_init__"):
                super().__post_init__()
            if not 0 <= self.stop < 1:
                raise ValueError("stop must be in [0, 1)")
            if self.trail and self.stop == 0:
                raise ValueError("trail needs a stop")

        def weights(self, closes):
            w = super().weights(closes)
            return apply_stop(w, closes, self.stop, self.trail, getattr(self, "every", None))

    Stopped.__name__ = Stopped.__qualname__ = f"{cls.__name__}Stop"
    return Stopped
