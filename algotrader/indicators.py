"""Reusable technical indicators (all causal: each value uses data up to that bar)."""
import pandas as pd

from .strategy import true_range


def wilder_mean(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (EMA with alpha = 1/period); NaN until `period` values."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. 100 if there are only gains, 0 if only losses, 50 if no change."""
    delta = close.diff()
    gain = wilder_mean(delta.clip(lower=0), period)
    loss = wilder_mean(-delta.clip(upper=0), period)
    out = 100 - 100 / (1 + gain / loss)
    return out.mask((gain == 0) & (loss == 0), 50.0)


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index."""
    up, down = bars["high"].diff(), -bars["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    atr = wilder_mean(true_range(bars), period)
    plus_di = 100 * wilder_mean(plus_dm, period) / atr
    minus_di = 100 * wilder_mean(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder_mean(dx, period)


def session_flags(index: pd.DatetimeIndex) -> tuple:
    """(first, last) boolean Series marking each calendar day's first and last bar."""
    day = pd.Series(index.normalize(), index=index)
    return day.ne(day.shift()), day.ne(day.shift(-1))


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """VWAP since each session's first bar, on the typical price (H+L+C)/3.

    NaN until the session has traded volume.
    """
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    day = bars.index.normalize()
    pv = (typical * bars["volume"]).groupby(day).cumsum()
    vol = bars["volume"].groupby(day).cumsum()
    return (pv / vol.where(vol > 0)).rename("vwap")
