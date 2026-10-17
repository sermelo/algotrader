import numpy as np
import pandas as pd
import pytest

from algotrader.movestats import Accumulator, normal_tail, window_changes
from tests.test_minutefeat import DAYS, frame, members


def test_window_changes_count_and_values_for_a_straight_line():
    got = dict(window_changes(frame(), members("AAA"), horizons=(10, 90, 240)))
    assert len(got[10]) == 2 * 301                                  # starts 30..330, both days, SPY is not a member
    assert len(got[90]) == 2 * 270                                  # the window must end by minute 389
    assert len(got[240]) == 2 * 120                                 # starts 30..149
    assert np.allclose(sorted(got[10])[0], 1.0 / (100 + 0.1 * 330 + 0.05), rtol=1e-3)      # smallest relative rise: the latest start
    assert np.allclose(sorted(got[10])[-1], 1.0 / (100 + 0.1 * 30 + 0.05), rtol=1e-3)


def test_price_floor_membership_and_early_close_windows():
    df = frame(tickers=("AAA", "BBB", "SPY"), base=100.0)
    assert len(dict(window_changes(df, members("AAA"), horizons=(10,)))[10]) == 2 * 301
    assert len(dict(window_changes(df, members("AAA", "BBB"), horizons=(10,)))[10]) == 2 * 301 * 2
    assert len(dict(window_changes(df, members("AAA"), horizons=(10,), min_price=1000.0))[10]) == 0
    assert len(dict(window_changes(df, members("AAA"), horizons=(10,), first_start=100, last_start=120))[10]) == 2 * 21


def test_accumulator_moments_tails_and_merge():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0001, 0.002, 200_000)
    a, b = Accumulator(), Accumulator()
    a.add(x[:100_000]); b.add(x[100_000:]); a.merge(b)
    s = a.summary()
    assert s["n"] == 200_000 and s["mean"] == pytest.approx(0.0001, abs=2e-5) and s["sd"] == pytest.approx(0.002, rel=0.01)
    assert abs(s["excess_kurtosis"]) < 0.1
    assert s["mean_abs"] == pytest.approx(0.002 * 0.7979, rel=0.02) and s["robust_sd"] == pytest.approx(0.002, rel=0.05)
    for t in s["tails"]:                                            # a normal variable matches the normal tails
        assert t["either"] == pytest.approx(t["normal_either"], abs=0.01)
    assert s["tails"][1]["up"] + s["tails"][1]["down"] == pytest.approx(s["tails"][1]["either"])
    assert a.hist.sum() == 200_000


def test_accumulator_counts_zeros_thresholds_and_fat_tails():
    a = Accumulator(thresholds=(0.001,))
    a.add(np.array([0.0, 0.0, 0.002, -0.002, 0.0005, np.nan]))
    s = a.summary()
    assert s["n"] == 5 and s["zero_share"] == pytest.approx(0.4)
    assert s["tails"][0]["up"] == pytest.approx(0.2) and s["tails"][0]["down"] == pytest.approx(0.2)
    rng = np.random.default_rng(1)
    t = Accumulator(); t.add(rng.standard_t(3, 100_000) * 0.001)
    assert t.summary()["excess_kurtosis"] > 1                       # fat tails show up
    assert t.summary()["sd"] > 1.2 * t.summary()["robust_sd"]       # and the plain spread is above the robust one
    assert normal_tail(0.002, 0.002) == pytest.approx(0.3173, abs=1e-3) and normal_tail(1, 0) == 0
