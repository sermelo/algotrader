"""Strategy definitions: indicators and signals."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    trend_filter: int = None

    def __post_init__(self):
        if not 0 < self.fast < self.slow:
            raise ValueError("require 0 < fast < slow")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return `bars` plus columns: sma_fast, sma_slow, atr, want_long."""
        df = bars.copy()
        close = df["close"]
        df["sma_fast"] = close.rolling(self.fast).mean()
        df["sma_slow"] = close.rolling(self.slow).mean()
        prev_close = close.shift()
        true_range = pd.concat([df["high"] - df["low"],
                                (df["high"] - prev_close).abs(),
                                (df["low"] - prev_close).abs()], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(self.atr_period).mean()
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
