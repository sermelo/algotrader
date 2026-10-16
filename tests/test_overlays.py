import math

import numpy as np
import pandas as pd
import pytest

from algotrader import Backtester, CostModel, PositionSizer
from algotrader.overlays import MarketRegime, VolTargeted, daily_closes, known_daily
from helpers import make_bars, make_days

FULL = PositionSizer(1.0, 1.0)


class AlwaysLong:
    atr_mult = 50.0

    def indicators(self, bars):
        df = bars.copy()
        df["atr"] = 1.0
        df["want_long"] = [i >= 1 for i in range(len(df))]
        return df


def alternating(n, step=0.01):
    """Daily closes alternating +step, -step: daily vol = step (about)."""
    closes, px = [], 100.0
    for i in range(n):
        px *= 1 + (step if i % 2 else -step)
        closes.append(px)
    return closes


def test_known_daily_uses_previous_session_for_intraday_bars():
    bars = make_days([[10, 11], [12, 13], [14, 15]])
    closes = daily_closes(bars)
    assert closes.tolist() == [11, 13, 15]
    mapped = known_daily(closes, bars)
    assert np.isnan(mapped.iloc[0]) and mapped.iloc[2] == 11 and mapped.iloc[5] == 13
    daily = make_bars([1, 2, 3], freq="1D")
    assert known_daily(daily_closes(daily), daily).tolist() == [1, 2, 3]


def test_vol_target_sizes_by_realised_volatility():
    bars = make_bars(alternating(60), freq="1D")
    ind = VolTargeted(AlwaysLong(), target_vol=0.08, lookback=20).indicators(bars)
    vol = ind["realised_vol"].iloc[-1]
    assert vol == pytest.approx(0.01 * math.sqrt(252), rel=0.05)
    assert ind["size_frac"].iloc[-1] == pytest.approx(0.08 / vol)
    assert ind["size_frac"].iloc[:20].isna().all()            # no vol estimate yet


def test_engine_caps_position_at_size_frac():
    bars = make_bars(alternating(60), freq="1D")
    strat = VolTargeted(AlwaysLong(), target_vol=0.08, lookback=20)
    t = Backtester(strat, CostModel(0, 0), FULL).run(bars, start=bars.index[30]).trades[0]
    frac = strat.indicators(bars)["size_frac"].iloc[30]
    assert t.shares == int(10_000 * frac // t.entry_price)
    assert t.shares * t.entry_price < 0.5 * 10_000


def test_market_regime_blocks_longs_below_market_average():
    bars = make_bars([10.0] * 30, freq="1D")
    market_up = make_bars(list(range(100, 130)), freq="1D")
    market_down = make_bars(list(range(130, 100, -1)), freq="1D")
    up = MarketRegime(AlwaysLong(), days=5).set_market(market_up).indicators(bars)
    down = MarketRegime(AlwaysLong(), days=5).set_market(market_down).indicators(bars)
    assert up["want_long"].iloc[10:].all() and not up["want_long"].iloc[:4].any()
    assert not down["want_long"].any()
    assert not MarketRegime(AlwaysLong(), days=5).indicators(bars)["want_long"].any()


def test_wrappers_validate_and_forward_fit():
    with pytest.raises(ValueError):
        VolTargeted(AlwaysLong(), target_vol=0)
    with pytest.raises(ValueError):
        MarketRegime(AlwaysLong(), days=0)

    class Fits(AlwaysLong):
        fitted = False

        def fit(self, bars):
            self.fitted = True
            return self

    inner = Fits()
    assert MarketRegime(VolTargeted(inner)).fit(None).atr_mult == 50.0 and inner.fitted
