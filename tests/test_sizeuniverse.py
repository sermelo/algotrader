import pandas as pd
import pytest

from algotrader.sizeuniverse import month_stats, rank_members, to_csv_text


def daily(rows):
    return pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def stats(**tickers):
    """month_stats-like frame: ticker=(dollar volume, days, price)."""
    s = pd.DataFrame({t: dict(dv=v[0], days=v[1], price=v[2]) for t, v in tickers.items()}).T
    return s


def test_month_stats_uses_median_dollar_volume_and_last_close():
    d = daily([("A", "2024-01-02", 10, 100), ("A", "2024-01-03", 10, 1000), ("A", "2024-01-04", 12, 100),
               ("B", "2024-01-03", 5, 10)])
    s = month_stats(d)
    assert s.loc["A", "dv"] == 1200 and s.loc["A", "days"] == 3
    assert s.loc["A", "price"] == 12 and s.loc["B", "days"] == 1
    assert s.attrs["date"] == pd.Timestamp("2024-01-04")


def test_ranks_by_dollar_volume_in_the_band():
    st = {pd.Timestamp("2024-01-31"): stats(A=(900, 20, 50), B=(500, 20, 50), C=(300, 20, 50), D=(100, 20, 50))}
    m = rank_members(st, lo=2, hi=3, window=1)
    assert sorted(m["ticker"]) == ["B", "C"]


def test_filters_price_days_and_excluded_names():
    st = {pd.Timestamp("2024-01-31"): stats(A=(900, 20, 50), CHEAP=(800, 20, 2), THIN=(700, 3, 50),
                                            ETF=(600, 20, 50), B=(500, 20, 50))}
    m = rank_members(st, lo=1, hi=10, window=1, min_price=5, min_days=10, exclude=["ETF"])
    assert sorted(m["ticker"]) == ["A", "B"]


def test_window_averages_months_and_a_new_listing_ranks_low():
    d1, d2 = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    st = {d1: stats(A=(100, 20, 50), B=(40, 20, 50)), d2: stats(A=(100, 20, 50), B=(40, 20, 50), NEW=(150, 20, 50))}
    top = rank_members(st, lo=1, hi=1, window=2)
    assert top[top["date"] == d2]["ticker"].tolist() == ["A"]            # NEW: (0 + 150) / 2 = 75 < 100
    assert rank_members(st, lo=1, hi=1, window=1).query("date == @d2")["ticker"].tolist() == ["NEW"]


def test_no_look_ahead_a_later_month_does_not_change_an_earlier_list():
    d1, d2 = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    a = {d1: stats(A=(100, 20, 50), B=(40, 20, 50))}
    b = a | {d2: stats(A=(1, 20, 50), B=(900, 20, 50))}
    pd.testing.assert_frame_equal(rank_members(a, lo=1, hi=1), rank_members(b, lo=1, hi=1).query("date == @d1").reset_index(drop=True))


def test_csv_round_trips_through_parse_members():
    from algotrader.history import parse_members
    m = pd.DataFrame({"date": pd.to_datetime(["2024-01-31"] * 2 + ["2024-02-29"]), "ticker": ["B", "A", "C"]})
    back = parse_members(to_csv_text(m))
    assert back["ticker"].tolist() == ["A", "B", "C"] and back["date"].nunique() == 2


def test_validation():
    with pytest.raises(ValueError):
        rank_members({}, lo=5, hi=2)
