import numpy as np
import pandas as pd
import pytest

from algotrader import CostModel
from algotrader.buffered import BufferedMomentum, buffered_picks
from algotrader.portfolio import PortfolioBacktester, panel
from algotrader.trend import MultiHorizonTrend
from helpers import make_bars


def frames(**closes):
    return {t: make_bars(c, freq="1D") for t, c in closes.items()}


def test_buffer_keeps_a_holding_until_it_leaves_keep_n():
    idx = pd.bdate_range("2024-01-01", periods=3)
    closes = pd.DataFrame(1.0, index=idx, columns=list("ABCD"))
    score = pd.DataFrame([[4, 3, 2, 1],      # day 0: buy A
                          [3, 4, 2, 1],      # day 1: A is 2nd, within keep_n=2: keep
                          [2, 4, 3, 1]],     # day 2: A is 3rd: replaced by B
                         index=idx, columns=list("ABCD"), dtype=float)
    held = buffered_picks(score, closes, top_n=1, keep_n=2, every="D")
    assert held.to_numpy().tolist() == [[True, False, False, False],
                                        [True, False, False, False],
                                        [False, True, False, False]]
    plain = buffered_picks(score, closes, top_n=1, keep_n=1, every="D")
    assert plain.iloc[1].tolist() == [False, True, False, False]   # no buffer: switches


def test_buffer_skips_names_not_trading_and_fills_slots():
    idx = pd.bdate_range("2024-01-01", periods=2)
    closes = pd.DataFrame({"A": [1.0, np.nan], "B": [1.0, 1.0], "C": [1.0, 1.0]}, index=idx)
    score = pd.DataFrame({"A": [3.0, 3.0], "B": [2.0, 2.0], "C": [1.0, 1.0]}, index=idx)
    held = buffered_picks(score, closes, top_n=2, keep_n=3, every="D")
    assert held.iloc[1].tolist() == [False, True, True]


def test_buffered_momentum_long_short_weights():
    up = [100 * 1.01 ** i for i in range(40)]
    flat = [100.0] * 40
    down = [100 * 0.99 ** i for i in range(40)]
    f = frames(A=up, B=flat, C=down)
    w = BufferedMomentum(lookback=10, top_n=1, keep_n=2, short_n=1, every="D").weights(
        panel(f, "close"))
    assert w.iloc[-1].to_dict() == {"A": 1.0, "B": 0.0, "C": -1.0}
    with pytest.raises(ValueError):
        BufferedMomentum(top_n=10, keep_n=5)


def test_trend_signal_and_vol_scaling():
    n = 300
    rng = np.random.default_rng(0)
    up = 100 * np.cumprod(1 + 0.003 + rng.normal(0, 0.002, n))
    down = 100 * np.cumprod(1 - 0.003 + rng.normal(0, 0.004, n))
    f = frames(SPY=list(up), TLT=list(down), AAPL=list(up))
    closes = panel(f, "close")
    s = MultiHorizonTrend(horizons=(21, 63), every="D").weights(closes)
    last = s.iloc[-1]
    assert last["SPY"] > 0 and last["TLT"] < 0
    assert last["AAPL"] == 0                          # not a trend asset
    assert abs(last["SPY"]) > abs(last["TLT"])        # lower vol -> bigger position
    lo = MultiHorizonTrend(horizons=(21, 63), every="D", long_only=True).weights(closes)
    assert lo.iloc[-1]["TLT"] == 0
    assert (s.iloc[:63] == 0).all().all()             # no signal before the longest horizon


def test_trend_runs_through_portfolio_engine_with_shorts():
    n = 300
    rng = np.random.default_rng(1)
    f = frames(SPY=list(100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))),
               GLD=list(100 * np.cumprod(1 + rng.normal(-0.001, 0.01, n))))
    res = PortfolioBacktester(MultiHorizonTrend(horizons=(21, 63)), CostModel(),
                              100_000, allow_short=True).run(f)
    assert res.equity.notna().all() and res.equity.iloc[-1] > 0


def test_trend_validation():
    with pytest.raises(ValueError):
        MultiHorizonTrend(horizons=())
    with pytest.raises(ValueError):
        MultiHorizonTrend(target_vol=0)
