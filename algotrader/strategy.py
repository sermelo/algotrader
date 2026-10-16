"""Strategy definitions: indicators and signals."""
from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    """What the Backtester needs from a strategy.

    `indicators(bars)` returns the bars plus at least a boolean `want_long`
    column (decided from information up to and including each bar's close) and
    an `atr` column used for the stop distance and sizing. `atr_mult` is the
    stop distance in ATRs.
    """
    atr_mult: float

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame: ...


def true_range(bars: pd.DataFrame) -> pd.Series:
    """max(high - low, |high - prev close|, |low - prev close|); first bar is high - low."""
    prev_close = bars["close"].shift()
    return pd.concat([bars["high"] - bars["low"],
                      (bars["high"] - prev_close).abs(),
                      (bars["low"] - prev_close).abs()], axis=1).max(axis=1)


def average_true_range(bars: pd.DataFrame, period: int) -> pd.Series:
    """Simple moving average of the true range over `period` bars."""
    return true_range(bars).rolling(period).mean()


@dataclass(frozen=True)
class SmaCrossStrategy:
    """Long-only SMA fast/slow crossover.

    The desired state ("in market") is True while SMA(fast) > SMA(slow). Signals
    are evaluated on bar close; the engine executes them at the NEXT bar's open,
    so there is no look-ahead.

    Attributes:
        fast: fast SMA window (bars).
        slow: slow SMA window (bars).
        atr_period: ATR window used for stop distance and position sizing.
        atr_mult: stop distance = atr_mult * ATR at entry.
        trend_filter: if set, only be in the market while close > SMA(trend_filter).
    """
    fast: int = 20
    slow: int = 50
    atr_period: int = 14
    atr_mult: float = 2.0
    trend_filter: int | None = None

    def __post_init__(self):
        if not 0 < self.fast < self.slow:
            raise ValueError("require 0 < fast < slow")
        if self.atr_period < 1 or not self.atr_mult > 0:
            raise ValueError("require atr_period >= 1 and atr_mult > 0")
        if self.trend_filter is not None and self.trend_filter < 1:
            raise ValueError("trend_filter must be a positive window or None")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return `bars` plus columns: sma_fast, sma_slow, atr, want_long."""
        df = bars.copy()
        close = df["close"]
        df["sma_fast"] = close.rolling(self.fast).mean()
        df["sma_slow"] = close.rolling(self.slow).mean()
        df["atr"] = average_true_range(df, self.atr_period)
        want = df["sma_fast"] > df["sma_slow"]
        if self.trend_filter:
            want &= close > close.rolling(self.trend_filter).mean()
        # NaN comparisons are False, so the warm-up period is automatically flat.
        df["want_long"] = want.fillna(False).astype(bool)
        return df

    @property
    def warmup(self) -> int:
        """Bars needed before the first valid signal."""
        return max(self.slow, self.atr_period, self.trend_filter or 0)
