"""Cross-sectional momentum with a buy/hold buffer (roadmap S1).

Novy-Marx & Velikov, "A taxonomy of anomalies and their trading costs", RFS 2016: the
most effective simple way to cut a strategy's trading costs is a buy/hold spread,
i.e. buy only names ranked in the top `top_n`, but keep holding a name until it drops
out of the top `keep_n` (> top_n). Turnover falls with little loss of the signal.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import _rebalance_days


def buffered_picks(score: pd.DataFrame, closes: pd.DataFrame, top_n: int, keep_n: int,
                   every: str, largest: bool = True) -> pd.DataFrame:
    """Boolean holdings per day: at each rebalance day keep held names still ranked
    within `keep_n` (and trading that day), then add the best-ranked others until
    `top_n` names are held; held unchanged between rebalance days."""
    rebal = _rebalance_days(closes.index, every)
    tradable = score.where(closes.notna())
    ranks = tradable.rank(axis=1, ascending=not largest, method="first")
    out = np.zeros(score.shape, dtype=bool)
    held = np.zeros(score.shape[1], dtype=bool)
    r = ranks.to_numpy()
    for i in range(len(score)):
        if rebal[i]:
            row = r[i]
            valid = ~np.isnan(row)
            keep = held & valid & (np.nan_to_num(row, nan=np.inf) <= keep_n)
            if keep.sum() > top_n:                          # keep the best top_n of them
                order = np.argsort(np.where(keep, row, np.inf))[:top_n]
                keep = np.zeros_like(keep)
                keep[order] = True
            need = top_n - keep.sum()
            if need > 0:
                cand = np.where(valid & ~keep, row, np.inf)
                add = np.argsort(cand)[:need]
                add = add[np.isfinite(cand[add])]
                keep[add] = True
            held = keep
        out[i] = held
    return pd.DataFrame(out, index=score.index, columns=score.columns)


@dataclass(frozen=True)
class BufferedMomentum:
    """Momentum (return from `lookback` to `skip` days ago) holding `top_n` names with a
    `keep_n` buffer; with `short_n` > 0 also short the bottom names with a `short_keep`
    buffer (default 2 x short_n). Equal weight per side, rebalanced at period ends."""
    lookback: int = 252
    skip: int = 0
    top_n: int = 10
    keep_n: int = 20
    short_n: int = 0
    short_keep: int = 0
    every: str = "M"

    def __post_init__(self):
        if self.lookback < 2 or not 0 <= self.skip < self.lookback or self.top_n < 1:
            raise ValueError("require lookback >= 2, 0 <= skip < lookback, top_n >= 1")
        if self.keep_n < self.top_n or self.short_n < 0:
            raise ValueError("require keep_n >= top_n and short_n >= 0")
        if self.short_n and self.short_keep and self.short_keep < self.short_n:
            raise ValueError("require short_keep >= short_n")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        c = closes.ffill()
        return c.shift(self.skip) / c.shift(self.lookback) - 1

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        score = self.score(closes)
        longs = buffered_picks(score, closes, self.top_n, self.keep_n, self.every, True)
        w = longs.div(longs.sum(axis=1).clip(lower=1), axis=0).astype(float)
        if self.short_n:
            keep = self.short_keep or 2 * self.short_n
            shorts = buffered_picks(score, closes, self.short_n, keep, self.every, False)
            w = w - shorts.div(shorts.sum(axis=1).clip(lower=1), axis=0).astype(float)
        return w
