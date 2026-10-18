from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from algotrader.costs import CostModel
from algotrader.portfolio import PortfolioBacktester, panel
from algotrader.stoploss import apply_stop, stop_events, with_stop
from helpers import make_bars

N = 91                                        # 2024-01-01 .. 2024-03-31, month ends at 30, 59, 90


def frames(**closes):
    return {t: make_bars(c, freq="1D", start="2024-01-01") for t, c in closes.items()}


def dip():
    a = [100.0] * 41 + [80.0] * 10 + [95.0] * 40                 # falls 20% on day 41, recovers
    return panel(frames(A=a, B=[100.0] * N), "close")


@dataclass(frozen=True)
class Half:
    """Half in A, half in B every day (a toy monthly strategy)."""
    every: str = "M"

    def weights(self, closes):
        return pd.DataFrame(0.5, index=closes.index, columns=closes.columns)


def test_fixed_stop_sells_until_the_next_rebalance_and_others_keep_their_drifted_weight():
    c = dip()
    w = Half().weights(c)
    out = apply_stop(w, c, 0.15, every="M")
    assert (out["A"].iloc[:41] == 0.5).all()
    assert (out["A"].iloc[41:59] == 0).all()                    # stopped on day 41's close
    assert out["B"].iloc[41] == pytest.approx(0.5 / 0.9)        # its value / the portfolio, not reset to 0.5
    assert (out.iloc[59:] == 0.5).all().all()                   # bought back at the Feb rebalance
    assert (apply_stop(w, c, 0.25, every="M") == w).all().all()  # a 20% fall does not reach a 25% stop


def test_stop_events_report_the_gain_given_up():
    c = dip()
    ev = stop_events(Half().weights(c), c, 0.15, every="M")
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["ticker"] == "A" and e["date"] == c.index[41]
    assert e["drop"] == pytest.approx(-0.2)
    assert e["after"] == pytest.approx(95 / 80 - 1)              # held to Feb 29 it would have gained this
    assert e["weight"] == pytest.approx(0.4 / 0.9)
    assert e["days_left"] == 18


def test_rebought_stock_gets_a_new_reference():
    a = [100.0] * 41 + [80.0] * 18 + [70.0] * 32                 # 70 is 12.5% under the new 80 reference
    c = panel(frames(A=a, B=[100.0] * N), "close")
    out = apply_stop(Half().weights(c), c, 0.15, every="M")
    assert (out["A"].iloc[59:] == 0.5).all()


def test_trailing_stop_uses_the_highest_close():
    a = [100.0 + 2 * i for i in range(11)] + [101.0] * (N - 11)  # up to 120, then back to 101
    c = panel(frames(A=a, B=[100.0] * N), "close")
    w = Half().weights(c)
    assert (apply_stop(w, c, 0.15, every="M") == w).all().all()   # 101 is not 15% under 100
    out = apply_stop(w, c, 0.15, trail=True, every="M")
    assert out["A"].iloc[11] == 0 and (out["A"].iloc[11:30] == 0).all()


def test_no_look_ahead():
    c = dip()
    w = Half().weights(c)
    out = apply_stop(w, c, 0.15, every="M")
    wrecked = c.copy()
    wrecked.iloc[46:] *= np.random.default_rng(0).uniform(0.3, 3.0, size=wrecked.iloc[46:].shape)
    out2 = apply_stop(w, wrecked, 0.15, every="M")
    pd.testing.assert_frame_equal(out.iloc[:46], out2.iloc[:46])


def test_through_the_engine_only_the_stopped_stock_is_sold():
    f = frames(A=[100.0] * 41 + [80.0] * 10 + [95.0] * 40, B=[100.0] * N)
    Stopped = with_stop(Half)
    res = PortfolioBacktester(Stopped(stop=0.15), CostModel(0.0, 0.0, 0.0, 0.0), 10_000.0).run(f)
    h = res.holdings
    assert h["A"].iloc[42] == 0 and h["A"].iloc[41] > 0         # sold at day 42's open
    assert abs(h["B"].iloc[42] - h["B"].iloc[41]) <= 1          # B left alone (at most a rounding share)
    assert h["A"].iloc[60] > 0                                  # back after the rebalance
    plain = PortfolioBacktester(Half(), CostModel(0.0, 0.0, 0.0, 0.0), 10_000.0).run(f)
    off = PortfolioBacktester(Stopped(), CostModel(0.0, 0.0, 0.0, 0.0), 10_000.0).run(f)
    pd.testing.assert_series_equal(plain.equity, off.equity)    # stop 0 is the plain strategy


def test_validation():
    Stopped = with_stop(Half)
    with pytest.raises(ValueError):
        Stopped(stop=1.0)
    with pytest.raises(ValueError):
        Stopped(stop=0.0, trail=True)
    assert Stopped.__name__ == "HalfStop"


def test_experiments_register_without_touching_the_other_fingerprints():
    from algotrader import longrun, longrun_stop
    before = {e.name: longrun.fingerprint(e) for e in longrun.EXPERIMENTS}
    assert len(longrun_stop.EXPERIMENTS) == 4
    for e in longrun_stop.EXPERIMENTS:
        assert len(e.combos()) == 11 and e.name not in before
        assert longrun_stop.fingerprint(e)
    assert {e.name: longrun.fingerprint(e) for e in longrun.EXPERIMENTS} == before
