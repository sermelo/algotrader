import numpy as np
import pandas as pd
import pytest

from algotrader.seasonal import SameMonthSeasonality


def test_picks_stock_that_rises_every_january():
    idx = pd.bdate_range("2008-01-01", "2014-01-31")
    a = np.ones(len(idx)); b = np.ones(len(idx))
    jan = idx.month == 1
    a[jan] = 1.01                     # A gains 1% a day each January
    b[~jan] = 1.0005                  # B gains outside January
    closes = pd.DataFrame({"A": 100 * np.cumprod(a), "B": 100 * np.cumprod(b)}, index=idx)
    st = SameMonthSeasonality(years=5, min_years=3, top_n=1)
    w = st.weights(closes)
    assert w.loc["2013-12-31"].to_dict() == {"A": 1.0, "B": 0.0}     # for January 2014
    assert w.loc["2013-11-29"].to_dict() == {"A": 0.0, "B": 1.0}     # for December
    # score uses only past years: before 3 Januaries are known, no January pick
    assert w.loc["2009-12-31"].abs().sum() == 0


def test_long_short_and_validation():
    idx = pd.bdate_range("2008-01-01", "2013-12-31")
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 4)), axis=0),
                          index=idx, columns=list("ABCD"))
    w = SameMonthSeasonality(years=5, min_years=3, top_n=1, short_n=1).weights(closes)
    last = w.iloc[-1]
    assert (last == 1).sum() == 1 and (last == -1).sum() == 1
    with pytest.raises(ValueError):
        SameMonthSeasonality(years=3, min_years=5)
