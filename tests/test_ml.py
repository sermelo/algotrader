import numpy as np
import pandas as pd
import pytest

from algotrader import Backtester, CostModel
from algotrader.ml import FEATURES, GradientBoostingStrategy, forward_label, ml_features
from helpers import make_bars, make_days


def test_forward_label_uses_threshold_and_horizon():
    y = forward_label(make_bars([100, 100.1, 101, 100, 99]), horizon=2, threshold=0.005)
    assert y.iloc[:3].tolist() == [1.0, 0.0, 0.0] and y.iloc[3:].isna().all()


def test_features_are_causal_and_complete():
    rng = np.random.default_rng(0)
    days = [list(100 + np.cumsum(rng.normal(0, 0.2, 13))) for _ in range(30)]
    bars = make_days(days)
    full, cut = ml_features(bars), ml_features(bars.iloc[:200])
    assert list(full.columns) == FEATURES
    pd.testing.assert_frame_equal(full.iloc[:200], cut)


def test_learns_planted_pattern_out_of_sample():
    # every third bar jumps 1%: the bar before a jump is predictable from minute of day
    days = []
    for _ in range(60):
        px, day = 100.0, []
        for i in range(13):
            px *= 1.01 if i % 3 == 2 else 0.997
            day.append(px)
        days.append(day)
    bars = make_days(days)
    split = bars.index[13 * 40]
    strat = GradientBoostingStrategy(horizon=1, threshold=0.005, p_enter=0.5, max_iter=50)
    strat.fit_many([bars[bars.index < split]])
    ind = strat.indicators(bars)
    later = ind[ind.index >= split]
    jump_next = (bars["close"].shift(-1) / bars["close"] - 1 > 0.005)[later.index]
    assert (later["want_long"] == jump_next).iloc[:-1].mean() > 0.95
    res = Backtester(strat, CostModel(0, 0)).run(bars, start=split)
    assert res.equity.iloc[-1] > res.initial_capital


def test_validation_and_fit_required():
    with pytest.raises(RuntimeError):
        GradientBoostingStrategy().indicators(make_bars([1, 2, 3]))
    for kwargs in ({"horizon": 0}, {"p_enter": 1.0}, {"max_depth": 0}):
        with pytest.raises(ValueError):
            GradientBoostingStrategy(**kwargs)
    with pytest.raises(ValueError):
        GradientBoostingStrategy(threshold=0.5).fit(make_bars([100 + 0.01 * i for i in range(50)]))
