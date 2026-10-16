"""Post-earnings-announcement drift with the market's reaction as the surprise (roadmap S7).

Chan, Jegadeesh & Lakonishok (1996) and Brandt, Kishore, Santa-Clara & Venkatachalam
(2008, "Earnings announcements are full of surprises"): the abnormal return over the
days around an earnings release predicts returns over the following months, more
strongly than the accounting surprise (SUE). The release time comes from the SEC 8-K
(item 2.02) acceptance time: released at or after 16:00 New York time, the first day
of reaction is the next trading day. The XBRL EPS values are filed later (10-Q/10-K),
so using them at the release date would be look-ahead; the price reaction is not.

Signal (daily panel): EAR = stock return - market return from the close before the
reaction day to the close one day after it (2 trading days), available from the next
day's close, kept for `hold_days` trading days.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import _top_weights

NY = "America/New_York"


def reaction_days(earnings: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """First trading day of reaction for each release (Series aligned to `earnings`)."""
    acc = pd.to_datetime(earnings["accepted"], utc=True, errors="coerce").dt.tz_convert(NY)
    filed = pd.to_datetime(earnings["filed"])
    day = acc.dt.tz_localize(None).dt.normalize().where(acc.notna(), filed)
    after_close = (acc.dt.hour >= 16).fillna(False)
    pos = index.searchsorted(day.to_numpy(), side="left")              # same or next trading day
    pos = np.where(after_close.to_numpy() & (pos < len(index)) &
                   (index[np.minimum(pos, len(index) - 1)] == day.to_numpy()), pos + 1, pos)
    out = pd.Series(pd.NaT, index=earnings.index, dtype="datetime64[ns]")
    ok = pos < len(index)
    out[ok] = index[pos[ok]]
    return out


def announcement_returns(earnings: pd.DataFrame, cik_map: pd.DataFrame, closes: pd.DataFrame,
                         market: pd.Series, hold_days: int = 63) -> pd.DataFrame:
    """Daily panel of the latest EAR per ticker, from the close of the day after the
    reaction day for `hold_days` trading days (NaN otherwise)."""
    index = closes.index
    rel = closes.pct_change(fill_method=None)
    mret = market.reindex(index).pct_change(fill_method=None)
    two = (1 + rel) * (1 + rel.shift()) - 1                          # day d-1 and d
    mtwo = (1 + mret) * (1 + mret.shift()) - 1
    tick_of = {}
    for t, c in cik_map["cik"].dropna().items():
        if t in closes.columns:
            tick_of.setdefault(int(c), []).append(t)
    e = earnings[earnings["cik"].isin(tick_of)].copy()
    e["day"] = reaction_days(e, index)
    e = e.dropna(subset=["day"]).sort_values("day").drop_duplicates(["cik", "day"])
    panel = pd.DataFrame(np.nan, index=index, columns=closes.columns)
    pos_of = {d: i for i, d in enumerate(index)}
    for _, r in e.iterrows():
        i = pos_of[r["day"]] + 1                                      # close of the day after
        if i >= len(index):
            continue
        for t in tick_of[int(r["cik"])]:
            ear = two.iat[i, closes.columns.get_loc(t)] - mtwo.iat[i]
            if np.isfinite(ear):
                panel.iloc[i:i + hold_days, closes.columns.get_loc(t)] = ear
    return panel


@dataclass(frozen=True)
class EarningsDrift:
    """Hold the `top_n` tickers with the highest recent earnings-announcement return
    (and short the `short_n` lowest); equal weight, rebalanced at period ends."""
    top_n: int = 30
    short_n: int = 0
    every: str = "M"
    hold_days: int = 63
    panels: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.top_n < 1 or self.short_n < 0 or self.hold_days < 1:
            raise ValueError("require top_n >= 1, short_n >= 0, hold_days >= 1")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        score = self.panels[f"ear_{self.hold_days}"].reindex(index=closes.index, columns=closes.columns)
        w = _top_weights(score, closes, self.top_n, self.every, largest=True)
        if self.short_n:
            w = w - _top_weights(score, closes, self.short_n, self.every, largest=False)
        return w
