"""Turn the unadjusted Hugging Face daily bars (scripts/hf_daily.py) into adjusted series.

The HF minute data is raw: no split or dividend adjustment. Quandl WIKI and Tiingo carry
the events (`split` factor, cash `dividend` on the ex-date), so a series is adjusted with
the events of a reference source (`adjust_frame`). For a ticker without a reference,
`detect_splits` finds splits from the prices themselves (an overnight gap close to 1/k
or k together with a matching jump in volume); it cannot find dividends.

Convention (the same as WIKI and Tiingo): the last price stays as traded and earlier
prices are scaled by the product, over the events after them, of
(1 / split) * (1 - dividend / previous raw close).
"""
import numpy as np
import pandas as pd

# Forward split ratios (new shares per old) and reverse ratios (1 new per k old) looked for.
FORWARD = (1.5, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20)
REVERSE = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 50, 100)


def align_events(index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    """Events (columns `split`, `dividend`, indexed by ex-date) on the rows of `index`.

    An event dated on a day the series has no bar is moved to the next bar; events after
    the last bar are dropped. Several events on one bar combine (splits multiply,
    dividends add)."""
    out = pd.DataFrame({"split": 1.0, "dividend": 0.0}, index=index)
    if events is None or len(events) == 0:
        return out
    ev = events.reindex(columns=["split", "dividend"])
    ev = ev.assign(split=ev["split"].fillna(1.0), dividend=ev["dividend"].fillna(0.0))
    ev = ev[(ev["split"] != 1.0) | (ev["dividend"] != 0.0)]
    pos = index.searchsorted(ev.index)
    keep = pos < len(index)
    ev, pos = ev[keep], pos[keep]
    for p, g in ev.groupby(pos):
        out.iloc[p, out.columns.get_loc("split")] = g["split"].prod()
        out.iloc[p, out.columns.get_loc("dividend")] = g["dividend"].sum()
    return out


def adjustment_factor(raw_close: pd.Series, split: pd.Series = None, dividend: pd.Series = None) -> pd.Series:
    """Multiplier for each row's raw prices: product of the events after the row."""
    split = pd.Series(1.0, index=raw_close.index) if split is None else split.fillna(1.0)
    dividend = pd.Series(0.0, index=raw_close.index) if dividend is None else dividend.fillna(0.0)
    m = ((1.0 / split) * (1.0 - dividend / raw_close.shift(1))).fillna(1.0)
    return m[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)


def adjust_frame(raw: pd.DataFrame, events: pd.DataFrame = None, dividends: bool = True) -> pd.DataFrame:
    """Adjusted open/high/low/close/volume of a raw daily frame (index = date, columns
    open, high, low, close, volume) using `events`. Volume is only split-adjusted."""
    ev = align_events(raw.index, events)
    div = ev["dividend"] if dividends else None
    f = adjustment_factor(raw["close"], ev["split"], div)
    out = raw.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = raw[c] * f
    shares = (1.0 / ev["split"])[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    out["volume"] = raw["volume"] / shares
    out["raw_close"] = raw["close"]
    return out


def detect_splits(raw: pd.DataFrame, gap_tol: float = 0.05, vol_window: int = 20,
                  vol_frac: float = 0.4, min_jump: float = 1.25, exact_tol: float = 0.015) -> pd.Series:
    """Splits found in a raw daily frame, as a Series of split factors indexed by the
    ex-date (7.0 for a 7-for-1, 0.1 for a 1-for-10 reverse split).

    A day qualifies when open / previous close is within `gap_tol` (relative) of 1/k (of k
    for a reverse split) and the volume level changed with it and stayed changed: the
    median volume of the `vol_window` days from the day on, against the median of the days
    before, is at least `vol_frac` x k (forward) or at most 1 / (`vol_frac` x k) (reverse),
    and either way moved by at least `min_jump` x. A gap within `exact_tol` of a ratio of 2
    or more needs no volume test (a news gap is rarely that exact). News gaps fail the
    volume test: their spike fades. Of several ratios the closest to the gap wins."""
    gap = raw["open"] / raw["close"].shift(1)
    vol = raw["volume"]
    med = vol.shift(1).rolling(vol_window, min_periods=max(5, vol_window // 2)).median()
    after = vol[::-1].rolling(vol_window, min_periods=5).median()[::-1]
    vr = after / med                       # volume from the day on, against the days before
    found = {}
    for t in raw.index[1:]:
        g, v = gap.at[t], vr.at[t]
        if not np.isfinite(g):
            continue
        best, err = None, gap_tol
        for k in FORWARD:
            e = abs(g * k - 1.0)
            if e < err and (e < exact_tol and k >= 2 or v >= max(vol_frac * k, min_jump)):
                best, err = float(k), e
        for k in REVERSE:
            e = abs(g / k - 1.0)
            if e < err and (e < exact_tol and k >= 2 or v <= min(1.0 / (vol_frac * k), 1.0 / min_jump)):
                best, err = 1.0 / k, e
        if best is not None:
            found[t] = best
    return pd.Series(found, dtype=float)
