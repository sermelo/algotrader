"""Combined-signal portfolio and a market trend filter (roadmap item 9), for the long history.

`CombinedSignal` averages the cross-sectional z-scores of several known anomalies (12-1 month
momentum, low volatility, nearness to the 52-week high, gross profitability) and holds the top
N (and shorts the bottom N). Each ingredient is computed from data up to the day, so the
blend is causal. A stock needs at least `min_components` of the ingredients to be scored.

`trend_days` (0 = off) keeps the weights only while the market (SPY) closed above its
`trend_days`-day mean, cash otherwise; `TrendMomentum` does the same for plain momentum. The
market closes come from `panels['market']` (taken from SPY before the point-in-time mask hides it).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import CrossSectionalMomentum, _top_weights

COMPONENTS = ("momentum", "low_vol", "high52", "profitability")


def zscores(x: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score of each row, clipped to +-clip (rows without spread are NaN)."""
    sd = x.std(axis=1).replace(0.0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(sd, axis=0).clip(-clip, clip)


def regime(market: pd.Series, days: int) -> pd.Series:
    """True on days the market closed above its `days`-day mean (that day included)."""
    mean = market.rolling(days).mean()
    return (market > mean) & mean.notna()


def apply_trend(w: pd.DataFrame, panels: dict, days: int) -> pd.DataFrame:
    """Zero the weights on days the market is not above its `days`-day mean."""
    if not days:
        return w
    on = regime(panels["market"], days).reindex(w.index).fillna(False).astype(float)
    return w.mul(on, axis=0)


@dataclass(frozen=True)
class CombinedSignal:
    top_n: int = 50
    short_n: int = 0
    every: str = "M"
    lookback: int = 252
    skip: int = 21
    vol_lookback: int = 252
    high_lookback: int = 252
    components: tuple = COMPONENTS
    min_components: int = 3
    trend_days: int = 0
    panels: dict = field(default=None, repr=False, compare=False)    # may hold 'gross_profitability'

    def __post_init__(self):
        if self.top_n < 1 or self.short_n < 0:
            raise ValueError("require top_n >= 1, short_n >= 0")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")
        if not set(self.components) <= set(COMPONENTS) or not self.components:
            raise ValueError(f"components must be a subset of {COMPONENTS}")
        if not 1 <= self.min_components <= len(self.components):
            raise ValueError("require 1 <= min_components <= len(components)")
        if self.trend_days < 0:
            raise ValueError("trend_days must be >= 0")

    def parts(self, closes: pd.DataFrame) -> dict:
        r = closes.ffill().pct_change(fill_method=None)
        out = {}
        if "momentum" in self.components:
            f = closes.ffill()
            out["momentum"] = f.shift(self.skip) / f.shift(self.lookback) - 1
        if "low_vol" in self.components:
            out["low_vol"] = -r.rolling(self.vol_lookback).std()
        if "high52" in self.components:
            out["high52"] = closes / closes.rolling(self.high_lookback).max()
        if "profitability" in self.components and self.panels and "gross_profitability" in self.panels:
            out["profitability"] = self.panels["gross_profitability"].reindex(
                index=closes.index, columns=closes.columns)
        return {k: zscores(v.where(closes.notna())) for k, v in out.items()}

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        parts = list(self.parts(closes).values())
        if len(parts) < self.min_components:
            return pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        stack = np.stack([p.to_numpy() for p in parts])
        n = np.isfinite(stack).sum(axis=0)
        mean = np.where(n >= self.min_components, np.nansum(stack, axis=0) / np.maximum(n, 1), np.nan)
        return pd.DataFrame(mean, index=closes.index, columns=closes.columns)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        score = self.score(closes)
        w = _top_weights(score, closes, self.top_n, self.every, largest=True)
        if self.short_n:
            w = w - _top_weights(score, closes, self.short_n, self.every, largest=False)
        return apply_trend(w, self.panels, self.trend_days)


@dataclass(frozen=True)
class TrendMomentum(CrossSectionalMomentum):
    """Momentum held only while the market is above its `trend_days`-day mean."""
    trend_days: int = 200
    panels: dict = field(default=None, repr=False, compare=False)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return apply_trend(super().weights(closes), self.panels, self.trend_days)


