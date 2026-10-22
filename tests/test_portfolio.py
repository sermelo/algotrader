import numpy as np
import pandas as pd
import pytest

from algotrader import CostModel
from algotrader.portfolio import (CrossSectionalMomentum, PortfolioBacktester, ShortTermReversal,
                                  panel)
from helpers import make_bars

NO_COST = CostModel(0, 0)


def frames(**closes):
    return {t: make_bars(c, freq="1D") for t, c in closes.items()}


class Fixed:
    """Weights given as a DataFrame-producing function of the closes."""
    def __init__(self, fn):
        self.fn = fn

    def weights(self, closes):
        return self.fn(closes)


def all_in(ticker):
    return Fixed(lambda c: pd.DataFrame({ticker: 1.0}, index=c.index))


def test_targets_trade_at_next_open_and_mark_at_close():
    f = frames(A=[10, 11, 12, 13], B=[20, 20, 20, 20])
    res = PortfolioBacktester(all_in("A"), NO_COST).run(f)
    first = res.orders[0]
    assert first[0] == f["A"].index[1] and first[1] == "A"
    assert first[3] == f["A"]["open"].iloc[1]            # day 2's open (= day 1's close, 10)
    assert first[2] == 1000                              # 10,000 / 10
    assert res.equity.iloc[-1] == pytest.approx(1000 * 13)


def test_no_orders_while_weights_are_unchanged():
    f = frames(A=[10, 12, 9, 15, 11, 14], B=[5] * 6)
    res = PortfolioBacktester(all_in("A"), NO_COST).run(f)
    days = [o[0] for o in res.orders]                    # buy on day 2, sell at data end
    assert days == [f["A"].index[1], f["A"].index[-1]]   # holdings drift, no re-trading


def test_rebalance_sells_before_buying():
    def switch(c):
        w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        w.iloc[:3, 0] = 1.0                              # A first, then B
        w.iloc[3:, 1] = 1.0
        return w
    f = frames(A=[10] * 6, B=[20] * 6)
    res = PortfolioBacktester(Fixed(switch), NO_COST).run(f)
    day4 = [o for o in res.orders if o[0] == f["A"].index[4]]
    assert [o[1] for o in day4] == ["A", "B"] and day4[0][2] == -1000 and day4[1][2] == 500


def test_commission_and_spread_reduce_equity():
    f = frames(A=[10] * 5)
    base = PortfolioBacktester(all_in("A"), NO_COST).run(f).equity.iloc[-1]
    costly = PortfolioBacktester(all_in("A"), CostModel(spread_bps=10)).run(f).equity.iloc[-1]
    assert costly < base == 10_000


def test_delisted_ticker_is_sold_at_its_last_close():
    f = frames(A=[10, 10, 12, 13], B=[5] * 8)
    res = PortfolioBacktester(all_in("A"), NO_COST).run(f)
    sale = [o for o in res.orders if o[2] < 0][0]
    assert sale[0] == f["A"].index[-1] and sale[3] == 13
    assert res.holdings["A"].iloc[-1] == 0
    assert res.equity.iloc[-1] == pytest.approx(1000 * 13)


def test_missing_bar_is_traded_at_the_next_bar():
    f = frames(A=[10, 10, 10, 10, 10], B=[5] * 5)
    f["A"] = f["A"].drop(f["A"].index[1])               # no bar on day 2
    res = PortfolioBacktester(all_in("A"), NO_COST).run(f)
    assert res.orders[0][0] == f["B"].index[2]


def test_no_look_ahead_prefix_of_equity_unchanged_by_future_data():
    f = frames(A=[10, 11, 9, 12, 13, 8, 15, 14], B=[20, 19, 21, 22, 18, 25, 24, 26])
    strat = CrossSectionalMomentum(lookback=2, skip=0, top_n=1, every="D")
    full = PortfolioBacktester(strat, NO_COST).run(f).equity
    cut = PortfolioBacktester(strat, NO_COST).run({t: b.iloc[:6] for t, b in f.items()}).equity
    pd.testing.assert_series_equal(full.iloc[:6], cut)


def test_momentum_holds_past_winners_reversal_past_losers():
    f = frames(A=[10, 11, 12, 13], B=[10, 10, 10, 10], C=[10, 9, 8, 7])
    closes = panel(f, "close")
    mom = CrossSectionalMomentum(lookback=2, skip=0, top_n=1, every="D").weights(closes)
    rev = ShortTermReversal(lookback=2, top_n=1, every="D").weights(closes)
    assert mom.iloc[-1].to_dict() == {"A": 1.0, "B": 0.0, "C": 0.0}
    assert rev.iloc[-1].to_dict() == {"A": 0.0, "B": 0.0, "C": 1.0}
    assert (mom.iloc[:2].sum(axis=1) == 0).all()          # no scores yet


def test_weights_held_between_monthly_rebalances():
    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    rng = np.random.default_rng(0)
    f = {}
    for t in "ABCD":
        b = make_bars(list(100 + np.cumsum(rng.normal(0, 1, len(idx)))), freq="1D")
        b.index = idx
        f[t] = b
    w = CrossSectionalMomentum(lookback=10, skip=0, top_n=2, every="M").weights(panel(f, "close"))
    changes = w.diff().abs().sum(axis=1) > 0
    assert set(w.index[changes].month) <= {1, 2, 3}
    assert changes.sum() <= 3                            # at most one change per month end
    assert (w.sum(axis=1).iloc[-1]) == pytest.approx(1.0)


def test_validation():
    with pytest.raises(ValueError):
        CrossSectionalMomentum(lookback=5, skip=5)
    with pytest.raises(ValueError):
        ShortTermReversal(every="Q")
    with pytest.raises(ValueError):
        PortfolioBacktester(all_in("A")).run({})


def short_a():
    return Fixed(lambda c: pd.DataFrame({"A": -1.0}, index=c.index))


def test_short_position_profits_when_price_falls():
    f = frames(A=[10, 10, 9, 8])
    res = PortfolioBacktester(short_a(), NO_COST, allow_short=True).run(f)
    assert res.orders[0][2] == -1000                     # sold short at day 2's open (10)
    assert res.equity.iloc[-2] == pytest.approx(10_000 + 1000 * (10 - 9))


def test_long_only_backtester_ignores_negative_weights():
    res = PortfolioBacktester(short_a(), NO_COST).run(frames(A=[10, 10, 9, 8]))
    assert res.orders == [] and (res.equity == 10_000).all()


def test_borrow_fee_is_charged_daily_on_short_value():
    f = frames(A=[10] * 6)
    base = PortfolioBacktester(short_a(), NO_COST, allow_short=True).run(f).equity
    fee = PortfolioBacktester(short_a(), NO_COST, allow_short=True, borrow_bps=252).run(f).equity
    # 252 bp a year = 1 bp a day on $10,000 short (A ends at its last close, so no fee then)
    assert (base - fee).iloc[2] == pytest.approx(2 * 1.0)


def test_gross_exposure_is_capped():
    f = frames(A=[10] * 4, B=[10] * 4)
    ls = Fixed(lambda c: pd.DataFrame({"A": 1.0, "B": -1.0}, index=c.index))
    res = PortfolioBacktester(ls, NO_COST, allow_short=True, max_gross=1.0).run(f)
    first = {o[1]: o[2] for o in res.orders if o[0] == f["A"].index[1]}
    assert first == {"A": 500, "B": -500}
    with pytest.raises(ValueError):
        PortfolioBacktester(ls, max_gross=0)


def test_delisted_short_is_covered_at_last_close():
    f = frames(A=[10, 10, 9], B=[5] * 6)
    res = PortfolioBacktester(short_a(), NO_COST, allow_short=True).run(f)
    cover = res.orders[-1]
    assert cover[1] == "A" and cover[2] == 1000 and cover[3] == 9
    assert res.equity.iloc[-1] == pytest.approx(10_000 + 1000)


def test_long_short_momentum_weights():
    f = frames(A=[10, 11, 12, 13], B=[10, 10, 10, 10], C=[10, 9, 8, 7])
    w = CrossSectionalMomentum(lookback=2, skip=0, top_n=1, short_n=1,
                               every="D").weights(panel(f, "close"))
    assert w.iloc[-1].to_dict() == {"A": 1.0, "B": 0.0, "C": -1.0}


def test_pairs_enter_on_divergence_and_exit_on_convergence():
    from algotrader.portfolio import PairsStrategy
    n = 40
    a = [100.0] * n
    b = [100.0 + (0.1 if i % 2 else -0.1) for i in range(n)]   # small noise
    b[30], b[31], b[32] = 110, 110, 100.05                     # B jumps (A cheap), then back
    f = frames(A=a, B=b)
    strat = PairsStrategy(pairs=(("A", "B"),), lookback=20, entry_z=2.0, exit_z=0.5)
    w = strat.weights(panel(f, "close"))
    assert w["A"].iloc[30] == 0.5 and w["B"].iloc[30] == -0.5  # long cheap A, short rich B
    assert w["A"].iloc[32] == 0 and w["B"].iloc[32] == 0       # back inside: flat
    assert (w.iloc[:30] == 0).all().all()
    missing = PairsStrategy(pairs=(("A", "ZZZ"),)).weights(panel(f, "close"))
    assert (missing == 0).all().all()
    with pytest.raises(ValueError):
        PairsStrategy(entry_z=1.0, exit_z=1.0)


def drift_frames(n=300, seed=0):
    """A drifts up steadily; B, C, D are noise. Ranking by momentum-like features
    should learn to prefer A."""
    rng = np.random.default_rng(seed)
    out = {"A": list(100 * np.cumprod(1 + 0.003 + rng.normal(0, 0.005, n)))}
    for t in "BCD":
        out[t] = list(100 * np.cumprod(1 + rng.normal(0, 0.005, n)))
    return frames(**out)


def test_xs_features_are_ranked_and_causal():
    from algotrader.portfolio import xs_features
    f = drift_frames()
    closes = panel(f, "close")
    full, cut = xs_features(closes), xs_features(closes.iloc[:150])
    for k in full:
        assert full[k].stack().between(0, 1).all()
        pd.testing.assert_frame_equal(full[k].iloc[:150], cut[k])


def test_xs_ml_learns_to_hold_the_drifting_ticker_out_of_sample():
    from algotrader.portfolio import CrossSectionalML
    f = drift_frames()
    split = f["A"].index[200]
    strat = CrossSectionalML(horizon=5, top_n=1, every="D", max_iter=50)
    strat.fit_many({t: b[b.index < split] for t, b in f.items()})
    w = strat.weights(panel(f, "close"))
    assert w.loc[w.index >= split, "A"].mean() > 0.8


def test_xs_ml_validation():
    from algotrader.portfolio import CrossSectionalML
    with pytest.raises(RuntimeError):
        CrossSectionalML().weights(panel(drift_frames(), "close"))
    with pytest.raises(ValueError):
        CrossSectionalML(horizon=0)
    with pytest.raises(ValueError):
        CrossSectionalML().fit_many({t: b.iloc[:10] for t, b in drift_frames().items()})


def test_cash_shortfall_scales_every_buy_by_the_same_factor():
    # the 1% half-spread makes two half-weight buys cost 10,100 against 10,000 in cash: both are
    # cut to 495 shares (before the fix the first got all 500 and the second only 490)
    f = frames(A=[10, 10, 10], B=[10, 10, 10])
    half = Fixed(lambda c: pd.DataFrame(0.5, index=c.index, columns=c.columns))
    res = PortfolioBacktester(half, CostModel(commission_per_share=0, min_commission=0,
                                              spread_bps=100)).run(f)
    assert sorted((t, q) for _, t, q, _, _ in res.orders if q > 0) == [("A", 495), ("B", 495)]


HASH_SCRIPT = """
import sys; sys.path.insert(0, "tests")
import numpy as np, pandas as pd
from helpers import make_bars
from algotrader import CostModel
from algotrader.portfolio import PortfolioBacktester
rng = np.random.default_rng(0)
f = {f"T{i:02d}": make_bars(10 * np.cumprod(1 + rng.normal(0, 0.02, 40)), freq="1D") for i in range(30)}
class W:
    def weights(self, c):
        return pd.DataFrame(np.where(np.arange(len(c))[:, None] % 10 < 5, 1 / 30, 0.0) + 0 * c.to_numpy(),
                            index=c.index, columns=c.columns)
res = PortfolioBacktester(W(), CostModel(spread_bps=20)).run(f)
print(repr(res.orders), repr(res.equity.round(6).tolist()))
"""


def test_result_does_not_depend_on_the_hash_seed():
    import os
    import subprocess
    import sys
    runs = {subprocess.run([sys.executable, "-c", HASH_SCRIPT], capture_output=True, text=True, check=True,
                           env=os.environ | {"PYTHONHASHSEED": seed}).stdout for seed in ("1", "2", "3")}
    assert len(runs) == 1
