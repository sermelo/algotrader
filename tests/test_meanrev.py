import pytest

from algotrader.meanrev import RsiReversionStrategy
from helpers import make_bars

CLOSES = [10, 11, 12, 13, 14, 15, 13, 11, 12, 14, 16]


def test_enters_on_low_rsi_and_exits_above_short_sma():
    # RSI(2) on bar 7 is 14.3 (< 20); SMA(3) exit first hit on bar 9 (14 > 12.33)
    ind = RsiReversionStrategy(entry=20, exit_sma=3, trend_sma=None).indicators(
        make_bars(CLOSES, freq="1D"))
    assert ind["rsi"].iloc[7] == pytest.approx(100 - 100 / (1 + 0.25 / 1.5))
    assert ind.index[ind["want_long"]].tolist() == ind.index[[7, 8]].tolist()


def test_trend_filter_blocks_entries_below_trend():
    ind = RsiReversionStrategy(entry=20, exit_sma=3, trend_sma=3).indicators(
        make_bars(CLOSES, freq="1D"))
    assert not ind["want_long"].any()


@pytest.mark.parametrize("kwargs", [{"entry": 0}, {"entry": 100}, {"rsi_period": 0},
                                    {"exit_sma": 0}, {"trend_sma": 0}])
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        RsiReversionStrategy(**kwargs)
