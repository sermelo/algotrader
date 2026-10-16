"""Short-term mean reversion (Connors-style RSI(2)), meant for daily bars."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import rsi
from .strategy import average_true_range


@dataclass(frozen=True)
class RsiReversionStrategy:
    """Buy short-term weakness inside an uptrend; sell on the bounce.

    Enter when RSI(`rsi_period`) < `entry` (and, if `trend_sma` is set, close >
    SMA(`trend_sma`)); exit when close > SMA(`exit_sma`). Decided on closes, filled
    at the next open. No stop by default (`atr_mult` 50), as in Connors' rules.
    """
    rsi_period: int = 2
    entry: float = 10.0
    exit_sma: int = 5
    trend_sma: int | None = 200
    atr_period: int = 14
    atr_mult: float = 50.0

    def __post_init__(self):
        if self.rsi_period < 1 or self.exit_sma < 1 or not 0 < self.entry < 100:
            raise ValueError("require rsi_period >= 1, exit_sma >= 1, 0 < entry < 100")
        if self.trend_sma is not None and self.trend_sma < 1:
            raise ValueError("trend_sma must be >= 1 or None")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        close = df["close"]
        df["rsi"] = rsi(close, self.rsi_period)
        df["sma_exit"] = close.rolling(self.exit_sma).mean()
        enter = df["rsi"] < self.entry
        if self.trend_sma:
            df["sma_trend"] = close.rolling(self.trend_sma).mean()
            enter &= close > df["sma_trend"]
        leave = (close > df["sma_exit"]).to_numpy()
        enter = enter.to_numpy()
        want = np.zeros(len(df), dtype=bool)
        state = False
        for i in range(len(df)):
            state = (not leave[i]) if state else bool(enter[i])
            want[i] = state
        df["want_long"] = want
        df["atr"] = average_true_range(df, self.atr_period)
        return df
