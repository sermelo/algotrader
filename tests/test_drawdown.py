import numpy as np
import pandas as pd
import pytest

from algotrader import drawdown as dd
from algotrader.stats import returns_stats


def test_max_drawdown_by_hand_and_against_returns_stats():
    r = np.array([0.10, -0.50, 0.20, 0.0])                # 1.1 -> 0.55 -> 0.66
    assert dd.max_drawdown(r) == pytest.approx(-0.5)
    assert dd.max_drawdown(np.array([-0.2, 0.1])) == pytest.approx(-0.2)   # a loss on day one counts
    x = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 500))
    assert dd.max_drawdown(x.to_numpy()) == pytest.approx(returns_stats(x)["max_drawdown"])


def test_bounds_ordered_deterministic_and_zero_without_losses():
    r = pd.Series(np.random.default_rng(2).normal(0.0003, 0.01, 1000))
    b = dd.drawdown_bounds(r)
    assert b[0.975] <= b[0.95] <= b[0.9] <= b["median"] <= 0
    assert b == dd.drawdown_bounds(r)                                     # seeded
    year = dd.drawdown_bounds(r, horizon=252)
    assert year[0.95] > b[0.95]                                           # a shorter window has shallower drawdowns
    assert dd.drawdown_bounds(pd.Series([0.001] * 100))[0.975] == 0.0
    assert np.isnan(dd.drawdown_bounds(pd.Series([0.01] * 30))[0.95])     # fewer than two blocks


def test_lucky_ordering_shows_a_deeper_bound_than_realised():
    # losses alternate with gains, so the realised drawdown is one day's loss; reordered blocks
    # can string several losing days together
    loss = 1 / 1.02 - 1                                                    # each pair ends where it started
    r = pd.Series([0.02, loss] * 300)
    b = dd.drawdown_bounds(r, block=1)
    assert dd.max_drawdown(r.to_numpy()) == pytest.approx(loss)
    assert b[0.95] < -0.05


def test_report_from_walk_forward_files(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=600)
    rng = np.random.default_rng(3)
    for name in ("a", "b"):
        pd.DataFrame({"base": rng.normal(0, 0.01, 600), "hold": rng.normal(0, 0.012, 600)},
                     index=idx).to_csv(tmp_path / f"{name}_wf_returns.csv")
    t = dd.table(dd.read_returns(tmp_path))
    assert list(t.index) == ["a", "b", "buy_and_hold"]
    text = dd.write_report(tmp_path).read_text()
    assert "`buy_and_hold`" in text and text.count("\n| `") == 3
