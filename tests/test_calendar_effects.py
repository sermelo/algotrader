import numpy as np
import pandas as pd
import pytest

from algotrader import Backtester, CostModel, PositionSizer
from algotrader.calendar_effects import PreHolidayStrategy, holiday_positions


def days_with_holiday():
    # Mon 30 Jun .. Tue 8 Jul 2025, without Fri 4 Jul (Independence Day)
    idx = pd.bdate_range("2025-06-30", "2025-07-08")
    return idx[idx != pd.Timestamp("2025-07-04")]


def test_holiday_positions():
    before, after = holiday_positions(days_with_holiday())
    d = list(days_with_holiday().strftime("%a %d"))
    assert before[d.index("Thu 03")] == 0 and before[d.index("Wed 02")] == 1
    assert after[d.index("Mon 07")] == 1 and after[d.index("Tue 08")] == 2
    assert before[-1] > 100                           # end of data: no next session


def test_holds_the_day_before_the_holiday():
    idx = days_with_holiday()
    p = np.arange(100.0, 100.0 + len(idx))
    bars = pd.DataFrame({"open": p, "high": p + 0.5, "low": p - 0.5, "close": p + 0.2,
                         "volume": 1e6}, index=idx)
    st = PreHolidayStrategy(days_before=1, days_after=0, atr_period=2)
    res = Backtester(st, CostModel(0, 0), PositionSizer(1.0, 1.0)).run(bars)
    [t] = res.trades
    assert t.entry_time == pd.Timestamp("2025-07-03")      # open of the pre-holiday day
    assert t.exit_time == pd.Timestamp("2025-07-07")       # next open after the holiday
    with pytest.raises(ValueError):
        PreHolidayStrategy(days_before=0)
