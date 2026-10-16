import math

import numpy as np
import pandas as pd
import pytest

from algotrader.stats import (alpha_beta, expected_max_sharpe, portfolio_returns,
                              returns_stats, sharpe, sharpe_interval)


def days(n):
    return pd.date_range("2024-01-01", periods=n, freq="1D")


def test_portfolio_returns_equal_weight():
    idx = days(3)
    r = portfolio_returns({"a": pd.Series([1.0, 1.1, 1.21], index=idx),
                           "b": pd.Series([1.0, 1.0, 1.0], index=idx)})
    assert r.tolist() == pytest.approx([0.05, 0.05])


def test_returns_stats():
    st = returns_stats(pd.Series([0.1, -0.5, 0.2]))
    assert st["return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert st["max_drawdown"] == pytest.approx(-0.5)
    assert math.isnan(returns_stats(pd.Series([], dtype=float))["sharpe"])


def test_sharpe_interval_contains_estimate_and_narrows_with_data():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    lo, hi = sharpe_interval(r)
    assert lo < sharpe(r) < hi
    lo2, hi2 = sharpe_interval(pd.Series(rng.normal(0.001, 0.01, 4000)))
    assert hi2 - lo2 < hi - lo
    assert math.isnan(sharpe_interval(r.iloc[:10])[0])


def test_alpha_beta_recovers_known_relation():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0, 0.01, 2000))
    r = 0.5 * b + 0.0004 + pd.Series(rng.normal(0, 0.001, 2000))
    ab = alpha_beta(r, b)
    assert ab["beta"] == pytest.approx(0.5, abs=0.01)
    assert ab["alpha"] == pytest.approx(0.0004 * 252, rel=0.1)
    assert ab["alpha_t"] > 5


def test_expected_max_sharpe_grows_with_trials_and_shrinks_with_time():
    assert expected_max_sharpe(1, 3) == 0
    assert expected_max_sharpe(100, 3) > expected_max_sharpe(10, 3) > 0
    assert expected_max_sharpe(100, 12) < expected_max_sharpe(100, 3)
    # known value: about 2.53 standard errors for 100 trials
    assert expected_max_sharpe(100, 1) == pytest.approx(2.53, abs=0.02)
