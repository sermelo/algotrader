"""Session-aware strategies: overnight holding and intraday (flat at the close).

All decisions use bar closes and fill at the next bar's open (engine rules). Times
are bar END times in minutes after midnight, so they work for any bar size that
divides the session: with 5-minute bars the bar starting 15:25 ends at 15:30.
Intraday strategies set `exit_at_close` on each session's last bar.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import session_flags, session_vwap
from .strategy import average_true_range


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def bar_minutes(index: pd.DatetimeIndex) -> int:
    """Typical bar length in minutes (median spacing inside sessions)."""
    first, _ = session_flags(index)
    gaps = pd.Series(index).diff().dt.total_seconds().div(60)[~first.to_numpy()]
    return int(round(gaps.median())) if len(gaps.dropna()) else 1


def bars_per_session(index: pd.DatetimeIndex) -> int:
    """Bars in a full 390-minute session at this bar size (1 for daily or longer bars)."""
    first, _ = session_flags(index)
    if first.all():
        return 1
    return int(np.ceil(390 / bar_minutes(index)))


def bar_end_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes after midnight at which each bar ends."""
    return index.hour * 60 + index.minute + bar_minutes(index)


def _per_day(values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Broadcast a Series indexed by session date to every bar of that date."""
    return pd.Series(values.reindex(index.normalize()).to_numpy(), index=index)


def previous_session_close(bars: pd.DataFrame) -> pd.Series:
    """For each bar, the last close of the PREVIOUS session (NaN on the first day)."""
    _, last = session_flags(bars.index)
    daily = bars["close"][last.to_numpy()]
    daily.index = daily.index.normalize()
    return _per_day(daily.shift(), bars.index)


def session_open(bars: pd.DataFrame) -> pd.Series:
    """For each bar, the open of its session's first bar."""
    first, _ = session_flags(bars.index)
    opens = bars["open"][first.to_numpy()]
    opens.index = opens.index.normalize()
    return _per_day(opens, bars.index)


@dataclass(frozen=True)
class OvernightStrategy:
    """Hold from the last bar's open to the next session's open.

    Signal on each session's second-to-last bar (fill at the last bar's open, i.e.
    one bar before the close); `want_long` is False on the last bar, so the
    position is sold at the next session's first open. Use 1-minute bars for a
    close-to-open hold. Optional trend filter: only when the close is above the
    mean of the previous `trend_days` session closes. `skip_weekdays` (0 = Monday ..
    4 = Friday) skips the nights that start on those days (e.g. (4,) skips weekends).
    """
    trend_days: int | None = None
    atr_period: int = 14
    atr_mult: float = 50.0
    skip_weekdays: tuple = ()

    def __post_init__(self):
        if self.trend_days is not None and self.trend_days < 1:
            raise ValueError("trend_days must be >= 1 or None")
        if any(d not in range(5) for d in self.skip_weekdays):
            raise ValueError("skip_weekdays must be in 0..4")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        _, last = session_flags(df.index)
        want = last.shift(-1, fill_value=False) & ~last
        if self.trend_days:
            daily = df["close"][last.to_numpy()]
            daily.index = daily.index.normalize()
            sma_prev = _per_day(daily.rolling(self.trend_days).mean().shift(), df.index)
            want &= df["close"] > sma_prev
        if self.skip_weekdays:
            want &= ~pd.Series(df.index.dayofweek, index=df.index).isin(self.skip_weekdays)
        df["atr"] = average_true_range(df, self.atr_period)
        df["want_long"] = want.to_numpy()
        return df


@dataclass(frozen=True)
class IntradayMomentumStrategy:
    """Gao, Han, Li & Zhou (2018), long-only: hold the last part of the session when
    the first half hour (from the previous close) was up.

    r1 = close at `first_end` / previous session close - 1. If r1 > `min_ret`
    (and, with `require_r12`, the half hour before `entry` was also up), signal at
    the bar ending at `entry`, fill at the next open, exit at the session close.
    """
    first_end: str = "10:00"
    entry: str = "15:30"
    min_ret: float = 0.0
    require_r12: bool = False
    atr_period: int = 14
    atr_mult: float = 50.0

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        _, last = session_flags(df.index)
        end = bar_end_minutes(df.index)
        t1, te = _minutes(self.first_end), _minutes(self.entry)

        def close_at(minute):
            px = df["close"][end == minute]
            return _per_day(pd.Series(px.to_numpy(), index=px.index.normalize()), df.index)

        r1 = close_at(t1) / previous_session_close(df) - 1
        cond = r1 > self.min_ret
        if self.require_r12:
            cond &= close_at(te) / close_at(te - 30) - 1 > 0
        df["r1"] = r1
        df["want_long"] = ((end >= te) & cond.fillna(False)).to_numpy()
        df["exit_at_close"] = last.to_numpy()
        df["atr"] = average_true_range(df, self.atr_period)
        return df


@dataclass(frozen=True)
class NoiseAreaStrategy:
    """Zarattini, Aziz & Barbon (2024) "Beat the Market", long-only.

    For each time of day, sigma = mean over the previous `lookback` sessions of
    |close / session open - 1| at that time. Upper band = max(session open,
    previous close) * (1 + band_mult * sigma). At decision bars (bar end time a
    multiple of `decision_minutes`, from `first_decision` on): go long if close >
    band; while long, stay only if close > max(band, session VWAP). Flat at the
    close.
    """
    lookback: int = 14
    band_mult: float = 1.0
    decision_minutes: int = 30
    first_decision: str = "10:00"
    atr_period: int = 14
    atr_mult: float = 50.0

    def __post_init__(self):
        if self.lookback < 1 or not self.band_mult > 0 or self.decision_minutes < 1:
            raise ValueError("require lookback >= 1, band_mult > 0, decision_minutes >= 1")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        first, last = session_flags(df.index)
        end = bar_end_minutes(df.index)
        day = df.index.normalize()
        opn = session_open(df)
        move = (df["close"] / opn - 1).abs()
        piv = pd.DataFrame({"day": day, "end": end, "move": move.to_numpy()}) \
            .pivot_table(index="day", columns="end", values="move", aggfunc="last")
        piv = piv.ffill(axis=1)                         # bars with no trades: carry the move
        sigma = piv.rolling(self.lookback, min_periods=self.lookback).mean().shift()
        sig = sigma.stack().reindex(pd.MultiIndex.from_arrays([day, end])).to_numpy()
        ref = np.fmax(opn.to_numpy(), previous_session_close(df).to_numpy())
        band = ref * (1 + self.band_mult * sig)
        vwap = session_vwap(df).to_numpy()
        close = df["close"].to_numpy()
        decide = (end % self.decision_minutes == 0) & (end >= _minutes(self.first_decision))
        first_np = first.to_numpy()
        want = np.zeros(len(df), dtype=bool)
        state = False
        for i in range(len(df)):
            if first_np[i]:
                state = False
            if decide[i] and not np.isnan(band[i]):
                stop = max(band[i], vwap[i]) if not np.isnan(vwap[i]) else band[i]
                state = close[i] > (stop if state else band[i])
            want[i] = state
        df["band"], df["vwap"] = band, vwap
        df["want_long"] = want
        df["exit_at_close"] = last.to_numpy()
        df["atr"] = average_true_range(df, self.atr_period)
        return df


@dataclass(frozen=True)
class OpeningRangeBreakoutStrategy:
    """Zarattini & Aziz (2023) opening-range breakout, long-only.

    If the opening range (first `or_minutes`) closes above its open, buy at the next
    open with the stop at the range low, and exit at the session close. Optional
    "stocks in play" filter: opening-range volume at least `min_rel_volume` times its
    mean over the previous `rv_lookback` sessions. Optional trend filter: only when
    the previous session's close is above the mean of the `trend_days` session
    closes before it. Optional `profit_r`: take profit at entry + profit_r x (entry -
    range low), the paper's 10R target (rarely reached intraday).
    """
    or_minutes: int = 5
    min_rel_volume: float = 0.0
    rv_lookback: int = 14
    trend_days: int | None = None
    atr_period: int = 14
    atr_mult: float = 50.0
    profit_r: float | None = None

    def __post_init__(self):
        if self.profit_r is not None and self.profit_r <= 0:
            raise ValueError("profit_r must be > 0 or None")
        if self.or_minutes < 1 or self.rv_lookback < 1 or self.min_rel_volume < 0:
            raise ValueError("require or_minutes >= 1, rv_lookback >= 1, min_rel_volume >= 0")
        if self.trend_days is not None and self.trend_days < 1:
            raise ValueError("trend_days must be >= 1 or None")

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        _, last = session_flags(df.index)
        end = bar_end_minutes(df.index)
        or_end = _minutes("09:30") + self.or_minutes
        day = df.index.normalize()
        in_or = end <= or_end
        g = df[in_or].groupby(day[in_or])
        rng = pd.DataFrame({"open": g["open"].first(), "close": g["close"].last(),
                            "low": g["low"].min(), "volume": g["volume"].sum()})
        # the range is complete only if some bar ends exactly at or_end
        rng["complete"] = pd.Series(end == or_end, index=df.index).groupby(day).any()
        rel_vol = rng["volume"] / rng["volume"].rolling(self.rv_lookback).mean().shift()
        ok = rng["complete"] & (rng["close"] > rng["open"])
        if self.min_rel_volume > 0:
            ok &= rel_vol >= self.min_rel_volume
        if self.trend_days:
            daily = df["close"][last.to_numpy()]
            daily.index = daily.index.normalize()
            prev = daily.shift()
            ok &= (prev > prev.rolling(self.trend_days).mean().shift()).reindex(rng.index).fillna(False)
        df["or_low"] = _per_day(rng["low"], df.index)
        df["rel_volume"] = _per_day(rel_vol, df.index)
        ok_bar = _per_day(ok.astype(float), df.index).fillna(0.0) > 0
        df["want_long"] = ((end >= or_end) & ok_bar).to_numpy()
        df["stop_price"] = df["or_low"]
        if self.profit_r is not None:
            df["target_r"] = float(self.profit_r)
        df["exit_at_close"] = last.to_numpy()
        df["atr"] = average_true_range(df, self.atr_period)
        return df

