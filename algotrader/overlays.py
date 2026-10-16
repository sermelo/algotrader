"""Wrappers that change how any strategy is sized or when it may enter.

Each wrapper satisfies the Strategy protocol itself and forwards `fit`/`fit_many`
to the wrapped strategy, so it works with every strategy and the experiment runner.
"""
import math

import numpy as np
import pandas as pd

from .indicators import session_flags
from .stats import TRADING_DAYS


def daily_closes(bars: pd.DataFrame) -> pd.Series:
    """Last close of each session, indexed by session date."""
    _, last = session_flags(bars.index)
    closes = bars["close"][last.to_numpy()]
    closes.index = closes.index.normalize()
    return closes


def known_daily(values: pd.Series, bars: pd.DataFrame) -> pd.Series:
    """Map a per-session series onto bars using only sessions finished by each bar's
    close: daily bars get their own session's value (known at that close), intraday
    bars the previous session's."""
    first, _ = session_flags(bars.index)
    daily_bars = bool(first.all())
    src = values if daily_bars else values.shift()
    return pd.Series(src.reindex(bars.index.normalize()).to_numpy(), index=bars.index)


class _Wrapper:
    def __init__(self, inner):
        self.inner = inner

    @property
    def atr_mult(self):
        return self.inner.atr_mult

    def fit(self, bars):
        self.inner.fit(bars)
        return self

    def fit_many(self, frames):
        self.inner.fit_many(frames)
        return self


class VolTargeted(_Wrapper):
    """Size each entry to a target annualised volatility.

    size_frac = min(max_frac, target_vol / realised vol), the realised vol being the
    standard deviation of the last `lookback` daily close-to-close returns (sessions
    finished by the signal bar's close), annualised with sqrt(252).
    """

    def __init__(self, inner, target_vol: float = 0.15, lookback: int = 20,
                 max_frac: float = 1.0):
        if not target_vol > 0 or lookback < 2 or not max_frac > 0:
            raise ValueError("require target_vol > 0, lookback >= 2, max_frac > 0")
        super().__init__(inner)
        self.target_vol, self.lookback, self.max_frac = target_vol, lookback, max_frac

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = self.inner.indicators(bars)
        closes = daily_closes(bars)
        vol = closes.pct_change(fill_method=None).rolling(self.lookback).std() \
            * math.sqrt(TRADING_DAYS)
        realised = known_daily(vol, bars)
        df["realised_vol"] = realised
        frac = (self.target_vol / realised).clip(upper=self.max_frac)
        df["size_frac"] = frac.where(realised > 0)
        return df


class MarketRegime(_Wrapper):
    """Only be long while a market series is above its `days`-session mean (entries
    are blocked and open positions are exited when it is not).

    `set_market(bars)` gives the market's bars (for example SPY at any bar size);
    the regime uses its session closes, known by each bar's close as for
    `VolTargeted`. Without market data the strategy never enters.
    """

    def __init__(self, inner, days: int = 200):
        if days < 1:
            raise ValueError("days must be >= 1")
        super().__init__(inner)
        self.days = days
        self.market = None

    def set_market(self, market_bars: pd.DataFrame):
        self.market = daily_closes(market_bars)
        return self

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = self.inner.indicators(bars)
        if self.market is None:
            ok = pd.Series(False, index=df.index)
        else:
            above = (self.market > self.market.rolling(self.days).mean()).astype(float)
            ok = known_daily(above.where(self.market.rolling(self.days).count() == self.days),
                             bars).fillna(0.0) > 0
        df["regime_ok"] = ok.to_numpy()
        df["want_long"] = df["want_long"].to_numpy(bool) & ok.to_numpy()
        return df
