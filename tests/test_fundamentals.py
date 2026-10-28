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


def _one(rows, start, end):
    idx = pd.bdate_range(start, end)
    return signals(pd.DataFrame(rows), pd.DataFrame({"cik": [1]}, index=["AAA"]), pd.DataFrame({"AAA": 10.0}, index=idx))


def test_equity_tag_chosen_per_period_not_per_company():
    # Qualcomm's case: StockholdersEquity until 2019, then only the tag including non-controlling interests
    inc = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    op = _one([fact(1, "StockholdersEquity", "2019-12-31", 50, "2020-02-01"),
               fact(1, inc, "2019-12-31", 80, "2020-02-01"),                     # both reported: the first tag
               fact(1, "OperatingIncomeLoss", "2019-12-31", 100, "2020-02-01", start="2019-01-01"),
               fact(1, inc, "2020-12-31", 400, "2021-02-01"),
               fact(1, "OperatingIncomeLoss", "2020-12-31", 120, "2021-02-01", start="2020-01-01")],
              "2020-03-02", "2021-03-31")["operating_profitability"]["AAA"]
    assert op["2020-03-02"] == pytest.approx(100 / 50)
    assert op["2021-02-02"] == pytest.approx(120 / 400)          # the old rule kept the 2019 equity: 120 / 50


def test_revenue_and_cost_tags_switch_per_period_and_gross_profit_preferred():
    rows = [fact(1, "Revenues", "2017-12-31", 1000, "2018-02-01", start="2017-01-01"),
            fact(1, "CostOfRevenue", "2017-12-31", 600, "2018-02-01", start="2017-01-01"),
            fact(1, "GrossProfit", "2017-12-31", 420, "2018-02-01", start="2017-01-01"),
            fact(1, "Assets", "2017-12-31", 2000, "2018-02-01"),
            # 2018: new revenue standard tags, no GrossProfit
            fact(1, "RevenueFromContractWithCustomerExcludingAssessedTax", "2018-12-31", 1500, "2019-02-01",
                 start="2018-01-01"),
            fact(1, "CostOfGoodsAndServicesSold", "2018-12-31", 700, "2019-02-01", start="2018-01-01"),
            fact(1, "Assets", "2018-12-31", 2000, "2019-02-01")]
    gp = _one(rows, "2018-03-01", "2019-03-29")["gross_profitability"]["AAA"]
    assert gp["2018-03-01"] == pytest.approx(420 / 2000)
    assert gp["2019-03-01"] == pytest.approx((1500 - 700) / 2000)   # the old rule kept 2017's gross profit


def test_ratio_parts_from_the_same_period_and_values_expire_after_18_months():
    op = _one([fact(1, "StockholdersEquity", "2012-12-31", 500, "2013-02-01"),
               fact(1, "OperatingIncomeLoss", "2012-12-31", 100, "2013-02-01", start="2012-01-01"),
               fact(1, "OperatingIncomeLoss", "2013-12-31", 300, "2014-02-01", start="2013-01-01")],   # no 2013 equity
              "2013-03-01", "2014-12-31")["operating_profitability"]["AAA"]
    assert op["2014-03-03"] == pytest.approx(100 / 500)           # not 2013 income over 2012 equity
    assert op["2014-07-02"] == pytest.approx(100 / 500)           # 548 days after the 2012 period end
    assert np.isnan(op["2014-07-03"])                             # expired: no stale value carried forward


def test_asset_growth_needs_the_previous_fiscal_year():
    g = _one([fact(1, "Assets", "2010-12-31", 1000, "2011-02-01"),
              fact(1, "Assets", "2012-12-31", 1500, "2013-02-01"),          # 2011 never reported
              fact(1, "Assets", "2013-12-31", 1800, "2014-02-01")],
             "2013-03-01", "2014-03-31")["asset_growth"]["AAA"]
    assert np.isnan(g["2013-03-01"])                              # not a two-year growth of 0.5
    assert g["2014-03-03"] == pytest.approx(1800 / 1500 - 1)
