"""Pre-holiday effect (roadmap S13): Ariel, "High stock returns before holidays",
Journal of Financial Economics 1990; Lakonishok & Smidt 1988. Daily bars.

A market holiday is a weekday without a session (seen as a gap of more than one business
day between consecutive bars). The position is held during the `days_before` trading
days up to and including the last session before the holiday, and the `days_after`
sessions after it: bought at the open of the first of those days and sold at the open
after the last. Like the turn of the month, this uses only which days have sessions (the
exchange calendar, known in advance), not prices; the last bar of the data has no next
session and is never a pre-holiday day.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import average_true_range


def holiday_positions(index: pd.DatetimeIndex) -> tuple:
    """(before, after): before = 0 on the last session before a holiday, 1 the session
    before it...; after = 1 on the first session after a holiday, 2 the next...; else large."""
    days = pd.DatetimeIndex(index.normalize())
    n = len(days)
    gap_after = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        gap_after[i] = np.busday_count(days[i].date(), days[i + 1].date()) > 1
    big = 10 ** 6
    before = np.full(n, big)
    after = np.full(n, big)
    nxt = None
    for i in range(n - 1, -1, -1):
        if gap_after[i]:
            nxt = i
        if nxt is not None:
            before[i] = nxt - i
    prev = None
    for i in range(n):
        if i > 0 and gap_after[i - 1]:
            prev = i
        if prev is not None:
            after[i] = i - prev + 1
    return before, after


@dataclass(frozen=True)
class PreHolidayStrategy:
    days_before: int = 1
    days_after: int = 0
    atr_period: int = 14
    atr_mult: float = 50.0

    def __post_init__(self):
        if self.days_before < 1 or self.days_after < 0:
            raise ValueError("require days_before >= 1 and days_after >= 0")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        before, after = holiday_positions(df.index)
        hold = (before < self.days_before) | (after <= self.days_after)
        # want_long on day t's close means "be long during day t+1"
        df["want_long"] = np.append(hold[1:], False)
        df["atr"] = average_true_range(df, self.atr_period)
        return df
