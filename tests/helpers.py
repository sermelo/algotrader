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
