import numpy as np
import pandas as pd
import pytest

from algotrader import Backtester, CostModel, DecisionTreeStrategy, tree_features
from algotrader.indicators import adx, rsi
from algotrader.tree import FEATURES, next_bar_up
from helpers import make_bars


def zigzag(n, start=100.0, drift=0.01, swing=0.5):
    """Slow uptrend with alternating up/down bars: after a down bar the next bar is up."""
    return [start + drift * i + (swing if i % 2 else -swing) for i in range(n)]


def random_walk(n, seed=0):
    rng = np.random.default_rng(seed)
    return list(100 + np.cumsum(rng.normal(0, 0.2, n)))


def test_rsi_extremes_and_warmup():
    up = rsi(pd.Series(np.arange(1.0, 31.0)))
    assert up.iloc[:14].isna().all() and up.iloc[14:].eq(100).all()
    assert rsi(pd.Series(np.arange(30.0, 0.0, -1))).iloc[14:].eq(0).all()
    assert rsi(pd.Series([5.0] * 30)).iloc[14:].eq(50).all()


def test_adx_is_100_in_a_clean_uptrend():
    # every bar's high and low rise by 1: all movement is upward
    assert adx(make_bars(range(10, 80)), 14).iloc[-1] == pytest.approx(100)


def test_feature_values():
    closes = [100 + i for i in range(40)]
    f = tree_features(make_bars(closes))
    assert list(f.columns) == FEATURES
    assert f["ret_1"].iloc[-1] == pytest.approx(139 / 138 - 1)
    assert f["ret_15"].iloc[-1] == pytest.approx(139 / 124 - 1)
    assert f["sma_ratio"].iloc[-1] == pytest.approx(np.mean(closes[-14:]) / 139)
    assert f["sma_corr"].iloc[-1] == pytest.approx(1.0)
    # constant volume: VWAP is the mean typical price; make_bars pads high/low symmetrically
    bars = make_bars(closes)
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    assert f["vwap_ratio"].iloc[-1] == pytest.approx(typical.iloc[-14:].mean() / 139)
    assert f.iloc[:14].isna().any(axis=1).all()          # warm-up rows are undefined


def test_features_do_not_look_ahead():
    bars = make_bars(random_walk(300))
    full, prefix = tree_features(bars), tree_features(bars.iloc[:200])
    pd.testing.assert_frame_equal(full.iloc[:200], prefix)


def test_label_is_next_close_higher():
    y = next_bar_up(make_bars([1, 2, 2, 1, 3]))
    assert y.iloc[:4].tolist() == [1.0, 0.0, 0.0, 1.0] and np.isnan(y.iloc[-1])


def test_learns_a_pattern_and_predicts_out_of_sample():
    bars = make_bars(zigzag(400))
    strat = DecisionTreeStrategy().fit(bars.iloc[:200])
    ind = strat.indicators(bars)
    later = ind.iloc[250:]
    pd.testing.assert_series_equal(later["want_long"], later["ret_1"] < 0, check_names=False)


def test_predictions_do_not_look_ahead():
    bars = make_bars(random_walk(400, seed=1))
    strat = DecisionTreeStrategy().fit(bars.iloc[:200])
    full = strat.indicators(bars)["want_long"]
    prefix = strat.indicators(bars.iloc[:300])["want_long"]
    pd.testing.assert_series_equal(full.iloc[:300], prefix)


def test_flat_during_warmup():
    bars = make_bars(zigzag(100))
    ind = DecisionTreeStrategy().fit(bars).indicators(bars)
    assert not ind["want_long"].iloc[:14].any()


def test_needs_fit_and_valid_parameters():
    with pytest.raises(RuntimeError):
        DecisionTreeStrategy().indicators(make_bars(zigzag(50)))
    for kwargs in ({"max_depth": 0}, {"atr_period": 0}, {"atr_mult": 0}):
        with pytest.raises(ValueError):
            DecisionTreeStrategy(**kwargs)
    with pytest.raises(ValueError):                      # only up moves: nothing to learn
        DecisionTreeStrategy().fit(make_bars(range(10, 80)))


def test_runs_through_backtester():
    bars = make_bars(zigzag(400))
    strat = DecisionTreeStrategy(atr_mult=50).fit(bars.iloc[:200])
    result = Backtester(strat, CostModel(0, 0)).run(bars.iloc[200:])
    assert len(result.trades) > 50
    assert result.equity.iloc[-1] == pytest.approx(
        result.initial_capital + sum(t.pnl for t in result.trades))
