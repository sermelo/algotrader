"""Shared builders for tests."""
import pandas as pd


def make_bars(closes, opens=None, pad=0.5, start="2024-01-02 09:30", freq="1min"):
    """Synthetic bars. open[i] defaults to close[i-1]; high/low are +/- `pad` around
    the open/close range, so ATR is about 2*pad + gaps."""
    closes = list(map(float, closes))
    opens = list(map(float, opens)) if opens is not None else [closes[0]] + closes[:-1]
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({
        "open": opens,
        "high": [max(o, c) + pad for o, c in zip(opens, closes)],
        "low": [min(o, c) - pad for o, c in zip(opens, closes)],
        "close": closes,
        "volume": 1000,
    }, index=idx)


def make_days(day_closes, freq_minutes=30, start_date="2024-01-02", first_opens=None,
              volumes=None, pad=0.1):
    """Bars for several sessions, each starting 09:30, `freq_minutes` apart.

    day_closes: list of per-day close lists. A bar's open is the previous close in the
    same day; the first bar opens at `first_opens[d]` (default: previous day's last
    close, or its own close on day one). volumes: optional per-day lists.
    """
    frames, prev = [], None
    for d, closes in enumerate(day_closes):
        date = pd.Timestamp(start_date) + pd.Timedelta(days=d)
        first_open = first_opens[d] if first_opens else (prev if prev is not None else closes[0])
        opens = [first_open] + list(closes[:-1])
        idx = pd.date_range(date + pd.Timedelta("9h30min"), periods=len(closes),
                            freq=f"{freq_minutes}min")
        vol = volumes[d] if volumes else [1000] * len(closes)
        frames.append(pd.DataFrame({
            "open": [float(x) for x in opens],
            "high": [max(o, c) + pad for o, c in zip(opens, closes)],
            "low": [min(o, c) - pad for o, c in zip(opens, closes)],
            "close": [float(x) for x in closes],
            "volume": vol}, index=idx))
        prev = closes[-1]
    return pd.concat(frames)
