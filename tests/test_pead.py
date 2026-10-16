import numpy as np
import pandas as pd
import pytest

from algotrader.pead import EarningsDrift, announcement_returns, reaction_days

IDX = pd.bdate_range("2013-01-07", periods=10)       # Mon 7 .. Fri 18 Jan 2013


def test_reaction_day_uses_acceptance_time():
    e = pd.DataFrame({"cik": [1, 1, 1], "filed": ["2013-01-08", "2013-01-08", "2013-01-12"],
                      "accepted": ["2013-01-08T12:00:00.000Z",     # 07:00 NY: same day
                                   "2013-01-08T21:05:00.000Z",     # 16:05 NY: next day
                                   "2013-01-12T14:00:00.000Z"]})   # Saturday: Monday
    d = reaction_days(e, IDX)
    assert list(d) == [pd.Timestamp("2013-01-08"), pd.Timestamp("2013-01-09"),
                       pd.Timestamp("2013-01-14")]


def test_announcement_return_is_known_only_after_the_window():
    closes = pd.DataFrame({"AAA": [100, 100, 110, 121, 121, 121, 121, 121, 121, 121.0]},
                          index=IDX)
    market = pd.Series(100.0, index=IDX)
    market.iloc[3:] = 101.0
    e = pd.DataFrame({"cik": [1], "filed": ["2013-01-08"], "accepted": ["2013-01-08T21:30:00.000Z"]})
    p = announcement_returns(e, pd.DataFrame({"cik": [1]}, index=["AAA"]), closes, market,
                             hold_days=3)
    # reaction day Wed 9th (+10%), next day +10%: EAR = 21% - 1% from the close of the 10th
    assert np.isnan(p.loc["2013-01-09", "AAA"])
    assert p.loc["2013-01-10", "AAA"] == pytest.approx(0.21 - 0.01)
    assert p.loc["2013-01-14", "AAA"] == pytest.approx(0.20)
    assert np.isnan(p.loc["2013-01-15", "AAA"])                    # hold_days = 3


def test_earnings_drift_weights():
    closes = pd.DataFrame(1.0, index=IDX, columns=list("ABC"))
    panel = pd.DataFrame({"A": 0.05, "B": np.nan, "C": -0.08}, index=IDX)
    st = EarningsDrift(top_n=1, short_n=1, every="D", hold_days=63, panels={"ear_63": panel})
    assert st.weights(closes).iloc[-1].to_dict() == {"A": 1.0, "B": 0.0, "C": -1.0}
    with pytest.raises(ValueError):
        EarningsDrift(hold_days=0)
