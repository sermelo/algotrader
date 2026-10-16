"""Loading and resampling OHLCV bars."""
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
SESSION_OPEN = "9h30min"          # US regular session open (data times are naive ET)


class BarLoader:
    """Reads HF Data Library parquet files (`<TICKER>_clean.parquet`).

    Files hold columns `datetime, Open, High, Low, Close, Volume, source`.
    Bars are returned indexed by `datetime` with lowercase columns
    `open, high, low, close, volume`, sorted and de-duplicated.

    Args:
        data_dir: directory containing `<TICKER>_clean.parquet` files.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def path(self, ticker: str) -> Path:
        return self.data_dir / f"{ticker}_clean.parquet"

    def tickers(self) -> list:
        """Tickers available in `data_dir`, sorted."""
        return sorted(p.name.replace("_clean.parquet", "")
                      for p in self.data_dir.glob("*_clean.parquet"))

    def load(self, ticker: str, timeframe: str | None = None) -> pd.DataFrame:
        """Load one ticker, optionally resampled to `timeframe` (e.g. '5min', '1D')."""
        path = self.path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"no data for {ticker}: {path} does not exist")
        raw = pd.read_parquet(path)
        return self.normalize(raw, timeframe)

    @staticmethod
    def normalize(raw: pd.DataFrame, timeframe: str | None = None) -> pd.DataFrame:
        """Clean a raw frame into indexed lowercase OHLCV, optionally resampled."""
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[list(_AGG)]
        if timeframe:
            df = BarLoader.resample(df, timeframe)
        return df

    @staticmethod
    def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate bars to `timeframe`, dropping empty periods (nights, weekends).

        Intraday bar sizes are anchored to the 09:30 session open, so a '1h' bar
        covers 09:30-10:30 rather than a half-hour 09:00-10:00 stub. Daily and
        longer bars use pandas' default calendar alignment.
        """
        step = to_offset(timeframe)
        if isinstance(step, Tick) and pd.Timedelta(step) < pd.Timedelta("1D"):
            resampler = df.resample(step, origin="start_day", offset=SESSION_OPEN)
        else:
            resampler = df.resample(step)
        return resampler.agg(_AGG).dropna(subset=["open"])
