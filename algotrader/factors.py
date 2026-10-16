"""More cross-sectional portfolio strategies (daily bars, for PortfolioBacktester).

Each has `weights(closes)` returning target weights per day from closes up to that
day, rebalanced at period ends (`every`: 'D', 'W' or 'M') and held in between.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import _rebalance_days, _top_weights

SECTOR_ETFS = ("XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB")


def _check_every(every):
    if every not in ("D", "W", "M"):
        raise ValueError("every must be 'D', 'W' or 'M'")


def _restrict(closes: pd.DataFrame, assets) -> pd.DataFrame:
    """Closes of `assets` only (all tickers if None); other columns become NaN."""
    if assets is None:
        return closes
    sub = closes.copy()
    sub.loc[:, ~closes.columns.isin(assets)] = np.nan
    return sub


@dataclass(frozen=True)
class LowVolatility:
    """Ang, Hodrick, Xing & Zhang (2006) / Frazzini & Pedersen (2014), long side: hold
    the `top_n` tickers with the lowest volatility of daily returns over `lookback`
    days, equal weight."""
    lookback: int = 63
    top_n: int = 10
    every: str = "M"

    def __post_init__(self):
        if self.lookback < 2 or self.top_n < 1:
            raise ValueError("require lookback >= 2, top_n >= 1")
        _check_every(self.every)

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        r = closes.ffill().pct_change(fill_method=None)
        return r.rolling(self.lookback).std()

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return _top_weights(self.score(closes), closes, self.top_n, self.every, largest=False)


@dataclass(frozen=True)
class High52Week:
    """George & Hwang (2004), long side: hold the `top_n` tickers whose close is
    nearest their highest close of the last `lookback` days."""
    lookback: int = 252
    top_n: int = 10
    every: str = "M"

    def __post_init__(self):
        if self.lookback < 2 or self.top_n < 1:
            raise ValueError("require lookback >= 2, top_n >= 1")
        _check_every(self.every)

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        c = closes.ffill()
        return c / c.rolling(self.lookback).max()

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return _top_weights(self.score(closes), closes, self.top_n, self.every, largest=True)


@dataclass(frozen=True)
class SectorMomentum:
    """Moskowitz & Grinblatt (1999) industry momentum with sector ETFs: hold the
    `top_n` sector ETFs with the highest `lookback`-day return (skipping the last
    `skip` days), equal weight. Other tickers in the universe are ignored."""
    lookback: int = 126
    skip: int = 0
    top_n: int = 3
    every: str = "M"
    assets: tuple = SECTOR_ETFS

    def __post_init__(self):
        if self.lookback < 2 or not 0 <= self.skip < self.lookback or self.top_n < 1:
            raise ValueError("require lookback >= 2, 0 <= skip < lookback, top_n >= 1")
        _check_every(self.every)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        sub = _restrict(closes, self.assets)
        c = sub.ffill()
        score = c.shift(self.skip) / c.shift(self.lookback) - 1
        return _top_weights(score, sub, self.top_n, self.every, largest=True)


@dataclass(frozen=True)
class DualMomentum:
    """Antonacci's dual momentum: rank the `risky` assets by `lookback`-day return and
    hold the best `top_n`, but only those with a positive return (absolute momentum);
    the unused share goes to `safe` (or stays in cash if `safe` is None)."""
    lookback: int = 252
    top_n: int = 1
    every: str = "M"
    risky: tuple = ("SPY", "QQQ", "IWM", "GLD")
    safe: str | None = "TLT"

    def __post_init__(self):
        if self.lookback < 2 or self.top_n < 1:
            raise ValueError("require lookback >= 2, top_n >= 1")
        _check_every(self.every)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        sub = _restrict(closes, self.risky)
        c = sub.ffill()
        ret = c / c.shift(self.lookback) - 1
        picks = _top_weights(ret, sub, self.top_n, "D", largest=True)
        w = picks.where(ret > 0, 0.0)                     # absolute momentum: losers -> safe
        if self.safe and self.safe in closes:
            scored = ret.notna().any(axis=1)
            w[self.safe] = (1 - w.sum(axis=1)).where(scored & closes[self.safe].notna(), 0.0)
        rebal = _rebalance_days(closes.index, self.every)
        w[~rebal] = np.nan
        w.iloc[0] = w.iloc[0].fillna(0.0)
        return w.ffill().fillna(0.0)
