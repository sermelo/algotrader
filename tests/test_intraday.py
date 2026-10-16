import numpy as np
import pandas as pd
import pytest

from algotrader import Backtester, CostModel, PositionSizer
from algotrader.intraday import (IntradayMomentumStrategy, NoiseAreaStrategy,
                                 OpeningRangeBreakoutStrategy, OvernightStrategy,
                                 bar_end_minutes, bars_per_session, previous_session_close)
from helpers import make_days

NO_COST = CostModel(0, 0)
FULL = PositionSizer(risk_per_trade=1.0, max_position_frac=1.0)
SESSION = 13                                     # 30-min bars 09:30 ... 15:30


def run(strategy, bars):
    return Backtester(strategy, NO_COST, FULL).run(bars)


def test_bar_end_minutes_and_previous_close():
    bars = make_days([[10, 11, 12], [13, 14, 15]])
    assert list(bar_end_minutes(bars.index)) == [600, 630, 660] * 2   # 10:00, 10:30, 11:00
    assert previous_session_close(bars).tolist()[3:] == [12.0] * 3
    assert previous_session_close(bars).iloc[:3].isna().all()
    assert bars_per_session(bars.index) == 13                   # 390 / 30
    assert bars_per_session(pd.date_range("2024-01-01", periods=5, freq="1D")) == 1


def test_overnight_buys_before_close_and_sells_next_open():
    bars = make_days([[10, 10, 10, 11], [12, 12, 12, 12], [12, 12, 12, 12]],
                     first_opens=[10, 11.5, 12.5])
    res = run(OvernightStrategy(atr_period=2), bars)
    t = res.trades[0]
    assert t.entry_time == bars.index[3] and t.entry_price == 10    # last bar's open (day 1)
    assert t.exit_time == bars.index[4] and t.exit_price == 11.5    # next session's open
    assert len(res.trades) == 3                                     # one per night (last forced)


def test_overnight_trend_filter():
    bars = make_days([[10] * 4, [10] * 4, [9] * 4, [9] * 4])
    ind = OvernightStrategy(trend_days=2).indicators(bars)
    assert not ind["want_long"].any()                    # never above the mean of prior closes
    with pytest.raises(ValueError):
        OvernightStrategy(trend_days=0)


def momentum_days(first_half_hour_close):
    flat = [100.0] * SESSION
    day2 = [first_half_hour_close] + [first_half_hour_close] * (SESSION - 2) + [103.0]
    return make_days([flat, day2])


def test_intraday_momentum_trades_last_half_hour_after_up_open():
    bars = momentum_days(101.0)
    ind = IntradayMomentumStrategy().indicators(bars)
    assert ind["r1"].iloc[-1] == pytest.approx(0.01)
    res = run(IntradayMomentumStrategy(), bars)
    t = res.trades[0]
    assert t.entry_time == bars.index[-1]                # the 15:30 bar's open
    assert t.exit_reason == "close" and t.exit_price == 103.0
    assert res.position.iloc[-1] == 0


def test_intraday_momentum_skips_down_open_and_threshold():
    assert run(IntradayMomentumStrategy(), momentum_days(99.0)).trades == []
    assert run(IntradayMomentumStrategy(min_ret=0.02), momentum_days(101.0)).trades == []


QUIET = [100.1, 99.9] * 6 + [100.0]                 # small moves around the open
BREAKOUT = [100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
FADE = [100.0, 101, 102, 103, 104, 105, 104, 100.5, 100.4, 100.3, 100.2, 100.1, 100.0]


def noise_days(third=BREAKOUT):
    return make_days([QUIET, QUIET, QUIET, third], first_opens=[100] * 4)


def test_noise_area_needs_history_then_rides_breakout_to_close():
    bars = noise_days()
    ind = NoiseAreaStrategy(lookback=2).indicators(bars)
    assert np.isnan(ind["band"].iloc[:2 * SESSION]).all()      # no band before 2 prior days
    trades = run(NoiseAreaStrategy(lookback=2), bars).trades
    assert len(trades) == 1
    # 10:00 close is 100 (inside the band); 10:30 close 101 breaks out -> fill at 10:30 open
    assert trades[0].entry_time == bars.index[3 * SESSION + 2]
    assert trades[0].exit_reason == "close" and trades[0].exit_price == 112


def test_noise_area_exits_when_price_falls_below_vwap():
    bars = noise_days(FADE)
    fade = run(NoiseAreaStrategy(lookback=2), bars).trades[0]
    assert fade.exit_reason == "signal"
    # 13:30 close (100.5) is below the session VWAP: exit at the next bar's open
    assert fade.exit_time == bars.index[3 * SESSION + 8]


def test_noise_area_band_ignores_future_days():
    bars = noise_days()
    full = NoiseAreaStrategy(lookback=2).indicators(bars)["band"]
    cut = NoiseAreaStrategy(lookback=2).indicators(bars.iloc[:3 * SESSION + 5])["band"]
    pd.testing.assert_series_equal(full.iloc[:3 * SESSION + 5], cut)


def orb_days(volumes=None):
    up = [101, 102, 103, 104, 104.5, 105]             # 5-min bars from 09:30
    down = [99, 98, 99, 100, 100, 100]
    stopped = [101, 100, 99, 98, 97, 96]              # range low 99.9, later trades below
    return make_days([up, down, stopped], freq_minutes=5, first_opens=[100, 100, 100],
                     volumes=volumes, pad=0.1)


def test_orb_long_after_up_range_with_stop_at_range_low():
    bars = orb_days()
    trades = run(OpeningRangeBreakoutStrategy(), bars).trades
    assert len(trades) == 2                              # day 2's range closed down
    assert trades[0].entry_time == bars.index[1] and trades[0].entry_price == 101
    assert trades[0].exit_reason == "close" and trades[0].exit_price == 105
    assert trades[1].exit_reason == "stop"
    assert trades[1].exit_price == pytest.approx(99.9)   # low of the first 5-min bar


def test_orb_relative_volume_filter():
    vols = [[1000] * 6, [1000] * 6, [5000] + [1000] * 5]
    trades = run(OpeningRangeBreakoutStrategy(min_rel_volume=2, rv_lookback=2), orb_days(vols)).trades
    assert len(trades) == 1 and trades[0].entry_time.day == 4      # only the high-volume day
    with pytest.raises(ValueError):
        OpeningRangeBreakoutStrategy(or_minutes=0)
