"""Same-calendar-month return seasonality (roadmap S4).

Heston & Sadka, "Seasonality in the cross-section of stock returns", JFE 2008;
Keloharju, Linnainmaa & Nyberg, "Return seasonalities", JF 2016: stocks with high
returns in a given calendar month in past years tend to have high returns in that
month again. At each month end, score = mean return of each stock in the coming
calendar month over the previous `years` years (at least `min_years` of them); hold
the `top_n` highest (and short the `short_n` lowest) for that month.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import _rebalance_days


@dataclass(frozen=True)
class SameMonthSeasonality:
    years: int = 10
    min_years: int = 5
    top_n: int = 50
    short_n: int = 0

    def __post_init__(self):
        if not 1 <= self.min_years <= self.years or self.top_n < 1 or self.short_n < 0:
            raise ValueError("require 1 <= min_years <= years, top_n >= 1, short_n >= 0")

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Score at each month end (indexed by that month's last trading day)."""
        month_end = closes.ffill().groupby(closes.index.to_period("M")).last()
        monthly = month_end.pct_change(fill_method=None)          # return of each month
        out = {}
        periods = list(monthly.index)
        for i, p in enumerate(periods):
            target = p + 1                                          # the coming month
            past = [target - 12 * k for k in range(1, self.years + 1)]
            rows = monthly.reindex([q for q in past if q in monthly.index and q <= p])
            n = rows.notna().sum()
            out[p] = rows.mean().where(n >= self.min_years)
        score = pd.DataFrame(out).T
        last_day = closes.groupby(closes.index.to_period("M")).apply(lambda g: g.index[-1])
        score.index = pd.DatetimeIndex([last_day[p] for p in score.index])
        return score

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        s = self.score(closes).reindex(closes.index)
        tradable = s.where(closes.notna())
        w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        rebal = _rebalance_days(closes.index, "M")
        for side, n, asc in ((1.0, self.top_n, False), (-1.0, self.short_n, True)):
            if not n:
                continue
            ranks = tradable.rank(axis=1, ascending=asc, method="first")
            chosen = (ranks <= n) & tradable.notna()
            part = side * chosen.div(chosen.sum(axis=1).clip(lower=1), axis=0)
            w = part if w.isna().all().all() else w + part
        w[~rebal] = np.nan
        w.iloc[0] = w.iloc[0].fillna(0.0)
        return w.ffill().fillna(0.0)
