"""Time-series momentum and calendar strategies for daily bars."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import average_true_range


@dataclass(frozen=True)
class TimeSeriesMomentumStrategy:
    """Moskowitz, Ooi & Pedersen (2012), long side: hold while the trailing return is
    positive.

    Signal = close[t - skip] / close[t - lookback] - 1 > `min_ret` (bars), decided on
    the close. `skip` > 0 leaves out the most recent bars (the paper's 12-month
    lookback is often used skipping the last month). Combine with the `vol_target`
    overlay for the paper's volatility scaling.
    """
    lookback: int = 252
    skip: int = 0
    min_ret: float = 0.0
    atr_period: int = 14
    atr_mult: float = 50.0

    def __post_init__(self):
        if self.lookback < 2 or not 0 <= self.skip < self.lookback:
            raise ValueError("require lookback >= 2 and 0 <= skip < lookback")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        close = df["close"]
        df["tsmom"] = close.shift(self.skip) / close.shift(self.lookback) - 1
        df["want_long"] = (df["tsmom"] > self.min_ret).to_numpy()
        df["atr"] = average_true_range(df, self.atr_period)
        return df


def month_positions(index: pd.DatetimeIndex) -> tuple:
    """(from_start, from_end) trading-day positions of each daily bar in its month:
    1, 2, ... from the first trading day and -1, -2, ... from the last.

    Uses which days of the month have bars, i.e. the exchange calendar, which is
    known in advance; prices are not used. The last month of the data may be
    incomplete, so its from_end is NaN (otherwise the data's last day would look
    like a month end).
    """
    month = pd.Series(index.to_period("M"), index=index)
    from_start = month.groupby(month).cumcount() + 1
    size = month.groupby(month).transform("size")
    from_end = (from_start - size - 1).astype(float)
    from_end[month == month.iloc[-1]] = np.nan
    return from_start.to_numpy(), from_end.to_numpy()


@dataclass(frozen=True)
class TurnOfMonthStrategy:
    """Lakonishok & Smidt (1988) turn-of-the-month effect, daily bars.

    Hold over the last `days_before` trading days of each month and the first
    `days_after` of the next: buy at the open of trading day -days_before and sell at
    the open of day days_after + 1 (so the position is held through the close of day
    days_after). `want_long` is therefore True on the closes of days
    -(days_before + 1) .. days_after - 1.
    """
    days_before: int = 1
    days_after: int = 3
    atr_period: int = 14
    atr_mult: float = 50.0

    def __post_init__(self):
        if self.days_before < 0 or self.days_after < 1:
            raise ValueError("require days_before >= 0 and days_after >= 1")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        start, end = month_positions(df.index)
        df["day_from_start"], df["day_from_end"] = start, end
        # with days_before 0 this is the last close only: buy at day 1's open
        df["want_long"] = (end >= -(self.days_before + 1)) | (start <= self.days_after - 1)
        df["atr"] = average_true_range(df, self.atr_period)
        return df
