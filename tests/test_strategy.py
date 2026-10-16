import numpy as np
import pytest

from algotrader import SmaCrossStrategy
from helpers import make_bars


def test_rejects_bad_windows():
    with pytest.raises(ValueError):
        SmaCrossStrategy(fast=50, slow=20)


def test_sma_values():
    ind = SmaCrossStrategy(fast=2, slow=3).indicators(make_bars([1, 2, 3, 4, 5]))
    assert ind["sma_fast"].tolist()[1:] == [1.5, 2.5, 3.5, 4.5]
    assert ind["sma_slow"].tolist()[2:] == [2.0, 3.0, 4.0]
    assert np.isnan(ind["sma_fast"].iloc[0]) and np.isnan(ind["sma_slow"].iloc[1])


def test_flat_during_warmup_and_long_in_uptrend():
    ind = SmaCrossStrategy(fast=2, slow=3).indicators(make_bars([1, 2, 3, 4, 5]))
    assert ind["want_long"].tolist() == [False, False, True, True, True]


def test_not_long_in_downtrend_or_flat_market():
    s = SmaCrossStrategy(fast=2, slow=3)
    assert not s.indicators(make_bars([5, 4, 3, 2, 1]))["want_long"].any()
    assert not s.indicators(make_bars([5] * 6))["want_long"].any()


def test_trend_filter_blocks_signals_below_long_sma():
    closes = [10, 10, 10, 10, 10, 10, 5, 6, 7]   # rebound, but still below SMA(6)
    plain = SmaCrossStrategy(fast=2, slow=3).indicators(make_bars(closes))
    filt = SmaCrossStrategy(fast=2, slow=3, trend_filter=6).indicators(make_bars(closes))
    assert plain["want_long"].iloc[-1]
    assert not filt["want_long"].iloc[-1]


def test_atr_is_mean_true_range():
    ind = SmaCrossStrategy(fast=2, slow=3, atr_period=2).indicators(
        make_bars([10, 10, 10], pad=0.5))
    assert ind["atr"].iloc[-1] == pytest.approx(1.0)   # high-low = 1.0 each bar


def test_warmup_property():
    assert SmaCrossStrategy(fast=20, slow=50, trend_filter=100).warmup == 100
