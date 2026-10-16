import numpy as np
import pandas as pd
import pytest

from algotrader.fundamentals import FundamentalSort, signals


def fact(cik, tag, end, value, filed, form="10-K", start=None):
    return {"cik": cik, "taxonomy": "us-gaap", "tag": tag, "unit": "USD",
            "start": pd.Timestamp(start) if start else pd.NaT, "end": pd.Timestamp(end),
            "value": value, "filed": pd.Timestamp(filed), "form": form, "fy": None, "fp": None,
            "accession": "x"}


def _facts():
    rows = [
        # company 1: equity 500 known from 2013-02-02; amended to 600 on 2013-05-01
        fact(1, "StockholdersEquity", "2012-12-31", 500, "2013-02-01"),
        fact(1, "StockholdersEquity", "2012-12-31", 600, "2013-05-01", form="10-K/A"),
        # a later 10-Q's comparative of an OLDER period must not replace the latest
        fact(1, "StockholdersEquity", "2013-03-31", 550, "2013-04-20", form="10-Q"),
        fact(1, "StockholdersEquity", "2011-12-31", 1, "2013-04-20", form="10-Q"),
        fact(1, "EntityCommonStockSharesOutstanding", "2013-01-15", 10, "2013-02-01"),
        fact(1, "Assets", "2011-12-31", 1000, "2013-02-01"),          # comparative
        fact(1, "Assets", "2012-12-31", 1200, "2013-02-01"),
        fact(1, "Revenues", "2012-12-31", 900, "2013-02-01", start="2012-01-01"),
        fact(1, "Revenues", "2012-12-31", 250, "2013-02-01", start="2012-10-01"),   # a quarter
        fact(1, "CostOfRevenue", "2012-12-31", 300, "2013-02-01", start="2012-01-01"),
        fact(1, "OperatingIncomeLoss", "2012-12-31", 100, "2013-02-01", start="2012-01-01"),
        # company 2: equity negative -> no book-to-market
        fact(2, "StockholdersEquity", "2012-12-31", -50, "2013-02-01"),
        fact(2, "EntityCommonStockSharesOutstanding", "2013-01-15", 10, "2013-02-01"),
    ]
    return pd.DataFrame(rows)


def test_signals_are_point_in_time():
    idx = pd.bdate_range("2013-01-28", "2013-06-28")
    raw = pd.DataFrame({"AAA": 10.0, "BBB": 5.0}, index=idx)
    cik = pd.DataFrame({"cik": [1, 2]}, index=["AAA", "BBB"])
    s = signals(_facts(), cik, raw)
    btm = s["book_to_market"]
    assert np.isnan(btm.loc["2013-02-01", "AAA"])                 # filed that day: not yet known
    assert btm.loc["2013-02-04", "AAA"] == pytest.approx(500 / (10 * 10))
    assert btm.loc["2013-04-22", "AAA"] == pytest.approx(550 / 100)   # newer quarter
    assert btm.loc["2013-05-02", "AAA"] == pytest.approx(550 / 100)   # amendment of an older period
    assert np.isnan(btm.loc["2013-03-01", "BBB"])                 # negative book equity
    assert s["asset_growth"].loc["2013-03-01", "AAA"] == pytest.approx(0.2)
    assert s["gross_profitability"].loc["2013-03-01", "AAA"] == pytest.approx((900 - 300) / 1200)
    assert s["operating_profitability"].loc["2013-03-01", "AAA"] == pytest.approx(100 / 500)
    assert s["operating_profitability"].loc["2013-05-02", "AAA"] == pytest.approx(100 / 600)  # amended equity


def test_amendment_of_latest_period_updates_from_its_filing_date():
    idx = pd.bdate_range("2013-01-28", "2013-06-28")
    raw = pd.DataFrame({"AAA": 10.0}, index=idx)
    f = _facts()
    f = f[~((f["tag"] == "StockholdersEquity") & (f["form"] == "10-Q"))]
    s = signals(f, pd.DataFrame({"cik": [1]}, index=["AAA"]), raw)
    assert s["book_to_market"].loc["2013-04-30", "AAA"] == pytest.approx(5.0)
    assert s["book_to_market"].loc["2013-05-02", "AAA"] == pytest.approx(6.0)


def test_fundamental_sort_long_short():
    idx = pd.bdate_range("2013-01-01", periods=3)
    closes = pd.DataFrame(1.0, index=idx, columns=list("ABC"))
    panel = pd.DataFrame([[0.1, 0.5, 0.9]] * 3, index=idx, columns=list("ABC"))
    st = FundamentalSort("asset_growth", top_n=1, short_n=1, largest=False, every="D",
                         panels={"asset_growth": panel})
    assert st.weights(closes).iloc[-1].to_dict() == {"A": 1.0, "B": 0.0, "C": -1.0}
    with pytest.raises(ValueError):
        FundamentalSort(top_n=0)
