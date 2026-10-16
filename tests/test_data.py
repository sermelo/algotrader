import pandas as pd
import pytest

from algotrader import BarLoader


def raw_frame():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:30",
                                    "2024-01-02 09:31", "2024-01-02 09:32"]),
        "Open": [2, 1, 20, 3], "High": [3, 2, 30, 4], "Low": [1, 0.5, 10, 2],
        "Close": [2.5, 1.5, 25, 3.5], "Volume": [10, 5, 7, 8], "source": "iex"})


class TestNormalize:
    def test_lowercase_columns_and_datetime_index(self):
        df = BarLoader.normalize(raw_frame())
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "datetime"

    def test_sorted_and_deduplicated_keeping_last(self):
        df = BarLoader.normalize(raw_frame())
        assert df.index.is_monotonic_increasing
        assert len(df) == 3
        assert df.loc["2024-01-02 09:31", "open"] == 20

    def test_resample_aggregates_ohlcv(self):
        df = BarLoader.normalize(raw_frame(), "2min")
        first = df.iloc[0]          # 09:30 + (deduped) 09:31
        assert (first.open, first.high, first.low, first.close, first.volume) == (1, 30, 0.5, 25, 12)

    def test_resample_drops_empty_periods(self):
        raw = raw_frame()
        raw.loc[3, "datetime"] = pd.Timestamp("2024-01-03 09:30")  # overnight gap
        df = BarLoader.normalize(raw, "1min")
        assert not df.isna().any().any()
        assert len(df) == 3

    def test_intraday_bars_anchored_to_session_open(self):
        idx = pd.date_range("2024-01-02 09:30", "2024-01-02 15:59", freq="1min")
        raw = pd.DataFrame({"datetime": idx, "Open": 1.0, "High": 1.0, "Low": 1.0,
                            "Close": 1.0, "Volume": 1})
        df = BarLoader.normalize(raw, "1h")
        assert df.index[0] == pd.Timestamp("2024-01-02 09:30")
        assert df["volume"].tolist() == [60] * 6 + [30]      # 09:30..15:30, last is half

    def test_daily_bars_keep_calendar_labels(self):
        df = BarLoader.normalize(raw_frame(), "1D")
        assert df.index[0] == pd.Timestamp("2024-01-02")


class TestLoader:
    def test_load_and_list_from_parquet(self, tmp_path):
        raw_frame().to_parquet(tmp_path / "ABC_clean.parquet")
        loader = BarLoader(tmp_path)
        assert loader.tickers() == ["ABC"]
        assert len(loader.load("ABC")) == 3

    def test_missing_ticker_names_the_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="XYZ"):
            BarLoader(tmp_path).load("XYZ")
