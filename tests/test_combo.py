import numpy as np
import pandas as pd
import pytest

from algotrader.combo import CombinedSignal, TrendMomentum, apply_trend, regime, zscores
from algotrader.portfolio import panel
from helpers import make_bars

N = 60


def frames(**closes):
    return {t: make_bars(c, freq="1D") for t, c in closes.items()}


def last(w):
    return {k: v for k, v in w.iloc[-1].to_dict().items() if v}


def closes_panel():
    up = [100 + i * 0.5 + (0.1 if i % 2 else 0) for i in range(N)]          # steady, near its high
    wild = [100 + (6 if i % 2 else 0) - i * 0.2 for i in range(N)]            # noisy, falling
    flat = [100 + (1 if i % 2 else 0) for i in range(N)]
    return panel(frames(A=up, B=wild, C=flat), "close")


SMALL = dict(lookback=40, skip=5, vol_lookback=20, high_lookback=40, min_components=2, every="D")


def test_zscores_have_zero_mean_and_clip():
    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0], [5.0] * 10])
    z = zscores(x, clip=2.0)
    assert z.iloc[0].max() == 2.0 and z.iloc[0].min() > -1
    assert z.iloc[1].isna().all()                                             # no spread, no score


def test_blend_picks_the_stock_good_on_all_three_price_signals():
    s = CombinedSignal(top_n=1, components=("momentum", "low_vol", "high52"), **SMALL)
    assert last(s.weights(closes_panel())) == {"A": 1.0}
    s2 = CombinedSignal(top_n=1, short_n=1, components=("momentum", "low_vol", "high52"), **SMALL)
    assert last(s2.weights(closes_panel())) == {"A": 1.0, "B": -1.0}


def test_each_ingredient_alone_ranks_by_its_own_measure():
    c = closes_panel()
    mom = CombinedSignal(top_n=1, components=("momentum",), min_components=1, lookback=40, skip=5, every="D")
    low = CombinedSignal(top_n=1, components=("low_vol",), min_components=1, vol_lookback=20, high_lookback=40, every="D")
    assert mom.score(c).iloc[-1].idxmax() == "A" and mom.score(c).iloc[-1].idxmin() == "B"
    assert low.score(c).iloc[-1].idxmin() == "B"                              # the wildest has the lowest score


def test_profitability_panel_changes_the_pick_and_is_needed_for_min_components():
    c = closes_panel()
    prof = pd.DataFrame({"A": 0.0, "B": 1.0, "C": 0.2}, index=c.index)
    only = dict(components=("profitability",), min_components=1, top_n=1, every="D")
    assert last(CombinedSignal(panels={"gross_profitability": prof}, **only).weights(c)) == {"B": 1.0}
    assert CombinedSignal(panels={}, **only).score(c).isna().all().all()      # ingredient missing: no score
    full = CombinedSignal(top_n=1, min_components=4, lookback=40, skip=5, vol_lookback=20, high_lookback=40, every="D")
    assert full.score(c).isna().all().all()                                   # needs all four, has three


def test_stock_with_too_few_ingredients_is_not_scored():
    c = closes_panel()
    prof = pd.DataFrame({"A": np.nan, "B": 1.0, "C": 2.0}, index=c.index)
    s = CombinedSignal(min_components=4, panels={"gross_profitability": prof}, lookback=40, skip=5,
                       vol_lookback=20, high_lookback=40, every="D")
    row = s.score(c).iloc[-1]
    assert np.isnan(row["A"]) and row[["B", "C"]].notna().all()


def test_regime_and_trend_filter_go_to_cash_when_the_market_falls():
    market = pd.Series([100.0] * 10 + list(np.linspace(100, 80, 10)),
                       index=pd.date_range("2024-01-02", periods=20, freq="D"))
    on = regime(market, 5)
    assert not on.iloc[:4].any()                                              # no mean yet
    assert not on.iloc[-1]
    w = pd.DataFrame(0.5, index=market.index, columns=["A", "B"])
    up = pd.Series(np.linspace(100, 120, 20), index=market.index)
    assert apply_trend(w, {"market": up}, 5).iloc[-1].sum() == 1.0
    assert apply_trend(w, {"market": market}, 5).iloc[-1].sum() == 0.0
    assert apply_trend(w, {}, 0).equals(w)                                    # off: nothing needed


def test_trend_momentum_is_cash_in_a_falling_market_and_holds_in_a_rising_one():
    c = closes_panel()
    rising = pd.Series(np.linspace(100, 150, N), index=c.index)
    falling = pd.Series(np.linspace(150, 100, N), index=c.index)
    args = dict(lookback=40, skip=5, top_n=1, every="D", trend_days=10)
    assert last(TrendMomentum(panels={"market": rising}, **args).weights(c)) == {"A": 1.0}
    assert not TrendMomentum(panels={"market": falling}, **args).weights(c).iloc[-1].any()


def test_no_look_ahead_weights_do_not_change_when_later_prices_change():
    c = closes_panel()
    market = pd.Series(np.linspace(100, 150, N), index=c.index)
    kw = dict(top_n=1, components=("momentum", "low_vol", "high52"), trend_days=10, **SMALL)
    full = CombinedSignal(panels={"market": market}, **kw).weights(c)
    cut = 45
    wrecked, m2 = c.copy(), market.copy()
    wrecked.iloc[cut:] = wrecked.iloc[cut:] * np.random.default_rng(0).uniform(0.2, 3, wrecked.iloc[cut:].shape)
    m2.iloc[cut:] = 1.0
    other = CombinedSignal(panels={"market": m2}, **kw).weights(wrecked)
    pd.testing.assert_frame_equal(full.iloc[:cut], other.iloc[:cut])


def test_validation():
    for bad in (dict(top_n=0), dict(short_n=-1), dict(every="Y"), dict(components=("nope",)),
                dict(components=()), dict(min_components=5), dict(trend_days=-1)):
        with pytest.raises(ValueError):
            CombinedSignal(**bad)


def test_experiments_register_without_touching_the_other_fingerprints():
    from algotrader import longrun, longrun_combo
    before = {e.name: longrun.fingerprint(e) for e in longrun.EXPERIMENTS}
    names = [e.name for e in longrun_combo.EXPERIMENTS]
    assert len(set(names)) == 5 and not set(names) & set(before)
    for e in longrun_combo.EXPERIMENTS:
        assert e.combos()
    assert {e.name: longrun.fingerprint(e) for e in longrun.EXPERIMENTS} == before
