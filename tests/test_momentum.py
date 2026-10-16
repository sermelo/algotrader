import numpy as np
import pandas as pd
import pytest

from algotrader import Backtester, CostModel, PositionSizer
from algotrader.momentum import TimeSeriesMomentumStrategy, TurnOfMonthStrategy, month_positions
from helpers import make_bars

FULL = PositionSizer(1.0, 1.0)


def test_tsmom_long_only_after_positive_lookback_return():
    closes = [10, 11, 12, 13, 12, 11, 10, 9, 10, 11]
    ind = TimeSeriesMomentumStrategy(lookback=3).indicators(make_bars(closes, freq="1D"))
    assert ind["tsmom"].iloc[3] == pytest.approx(13 / 10 - 1)
    # bar 6: 10 / 13 - 1 < 0 -> flat; bar 9: 11 / 10 - 1 > 0 (vs bar 6) -> long
    assert ind["want_long"].tolist() == [False, False, False, True, True, False, False,
                                         False, False, True]


def test_tsmom_skip_ignores_recent_bars():
    closes = [10, 12, 14, 9]                      # last bar falls, but skip=1 ignores it
    ind = TimeSeriesMomentumStrategy(lookback=3, skip=1).indicators(make_bars(closes, freq="1D"))
    assert ind["tsmom"].iloc[3] == pytest.approx(14 / 10 - 1) and ind["want_long"].iloc[3]
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(lookback=3, skip=3)


def month_bars():
    idx = pd.bdate_range("2024-01-24", "2024-02-09")     # late Jan to early Feb
    bars = make_bars(list(range(100, 100 + len(idx))), freq="1D")
    bars.index = idx
    return bars


def test_month_positions():
    start, end = month_positions(month_bars().index)
    jan = month_bars().index.month == 1
    assert end[jan][-1] == -1 and end[jan][-2] == -2
    assert list(start[~jan][:3]) == [1, 2, 3]
    assert np.isnan(end[~jan]).all()          # last month of data may be incomplete


def test_turn_of_month_holds_last_day_through_third_day():
    bars = month_bars()
    res = Backtester(TurnOfMonthStrategy(days_before=1, days_after=3, atr_period=2), CostModel(0, 0), FULL,
                     ).run(bars)
    t = res.trades[0]
    assert t.entry_time == pd.Timestamp("2024-01-31")      # open of January's last day
    assert t.exit_time == pd.Timestamp("2024-02-06")       # open of Feb's 4th trading day
    assert len(res.trades) == 1


def test_turn_of_month_zero_days_before_enters_on_first_day():
    res = Backtester(TurnOfMonthStrategy(days_before=0, days_after=1, atr_period=2), CostModel(0, 0), FULL
                     ).run(month_bars())
    t = res.trades[0]
    assert t.entry_time == pd.Timestamp("2024-02-01") and t.exit_time == pd.Timestamp("2024-02-02")
    with pytest.raises(ValueError):
        TurnOfMonthStrategy(days_after=0)
