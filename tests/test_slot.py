import numpy as np
import pandas as pd
import pytest

from algotrader.costs import CostModel, ibkr_tiered
from algotrader.slot import SlotTrader, breakeven

FREE = CostModel(commission_per_share=0.0, min_commission=0.0, max_commission_frac=0.0)


def day_bars(opens, closes=None, date="2024-01-02", start="09:30"):
    closes = closes if closes is not None else opens
    idx = pd.date_range(f"{date} {start}", periods=len(opens), freq="1min")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + .01,
                         "low": np.minimum(opens, closes) - .01, "close": closes, "volume": 1000.0}, index=idx)


def edges_at(bars, when, **values):
    idx = pd.DatetimeIndex([bars.index[0] + pd.Timedelta(minutes=when)])
    return pd.DataFrame({t: [v] for t, v in values.items()}, index=idx)


def test_breakeven_matches_hand_calculation():
    fixed = CostModel(spread_bps=2.0)
    # 10 shares at $100: $1 minimum per order twice = $2 on $1,000, plus 4 bp of spread
    assert breakeven(fixed, 100.0, 1000.0) == pytest.approx(0.002 + 0.0004)
    tiered = CostModel(commission_per_share=0.0035, min_commission=0.35, max_commission_frac=0.01)
    assert breakeven(tiered, 125.0, 500.0) == pytest.approx(0.70 / 500)                  # 4 shares: minimum applies
    assert breakeven(tiered, 2.5, 500.0) == pytest.approx(1.40 / 500)                    # 200 shares: per-share fee
    assert breakeven(fixed, 100.0, 1000.0, extra_bps=3.0) == pytest.approx(0.0027)
    assert np.isnan(breakeven(fixed, 100.0, 50.0))                                       # not one share


def test_breakeven_with_pass_through_fees_is_higher_than_commission_alone():
    plain = CostModel(commission_per_share=0.0035, min_commission=0.35, max_commission_frac=0.01)
    assert breakeven(ibkr_tiered(), 125.0, 500.0) > breakeven(plain, 125.0, 500.0)


def test_buys_highest_edge_next_open_and_sells_after_the_horizon():
    a = day_bars([100.0] * 60, [100.0] * 60)
    b = day_bars([50.0] * 10 + [50.0, 51.0, 52.0] + [52.0] * 47)       # open rises after the entry
    bars = {"A": a, "B": b}
    e = edges_at(a, 5, A=0.01, B=0.03)
    res = SlotTrader(capital=1000, cost=FREE, hold_minutes=10).run(bars, e)
    t = res.trades.iloc[0]
    assert t["ticker"] == "B" and t["decision"] == b.index[5] and t["entry"] == b.index[6]
    assert t["exit"] == b.index[16] and t["buy"] == 50.0 and t["shares"] == 20


def test_edge_below_the_cost_hurdle_is_not_traded():
    a = day_bars([100.0] * 60)
    cost = CostModel(spread_bps=2.0)                                       # hurdle 0.24% at $1,000
    low = SlotTrader(capital=1000, cost=cost).run({"A": a}, edges_at(a, 5, A=0.002))
    high = SlotTrader(capital=1000, cost=cost).run({"A": a}, edges_at(a, 5, A=0.004))
    assert low.trades.empty and len(high.trades) == 1
    assert SlotTrader(capital=1000, cost=cost, min_margin=0.002).run({"A": a}, edges_at(a, 5, A=0.004)).trades.empty


def test_one_trade_a_day_and_the_next_day_uses_the_new_equity():
    d1 = day_bars([100.0] * 40 + [110.0] * 20, date="2024-01-02")
    d2 = day_bars([110.0] * 60, date="2024-01-03")
    a = pd.concat([d1, d2])
    idx = [d1.index[5], d1.index[8], d2.index[5]]
    e = pd.DataFrame({"A": [0.05, 0.05, 0.05]}, index=pd.DatetimeIndex(idx))
    res = SlotTrader(capital=1000, cost=FREE, hold_minutes=50).run({"A": a}, e)
    assert len(res.trades) == 2                                            # second signal on day one ignored
    assert res.trades["decision"].tolist() == [d1.index[5], d2.index[5]]
    assert res.trades["net"].iloc[0] == pytest.approx(10 * 10.0)           # 10 shares at 100, sold at 110
    assert res.trades["shares"].iloc[1] == 10 and res.equity.iloc[0] == pytest.approx(1100.0)   # $1,100 buys 10 at 110


def test_exits_at_the_session_close_when_the_horizon_does_not_fit():
    a = day_bars([100.0] * 30, [100.0] * 29 + [103.0])
    res = SlotTrader(capital=1000, cost=FREE, hold_minutes=120).run({"A": a}, edges_at(a, 5, A=0.05))
    t = res.trades.iloc[0]
    assert t["reason"] == "close" and t["exit"] == a.index[-1] and t["sell"] == 103.0
    assert t["net"] == pytest.approx(30.0)


def test_costs_come_off_the_result_and_whole_shares_only():
    a = day_bars([100.0] * 60)
    cost = CostModel(spread_bps=2.0)
    res = SlotTrader(capital=1000, cost=cost, hold_minutes=10).run({"A": a}, edges_at(a, 5, A=0.01))
    t = res.trades.iloc[0]
    assert t["shares"] == 9                                                # $1 commission and the spread leave room for 9, not 10
    expected = 9 * (100 * (1 - 0.0002) - 100 * (1 + 0.0002)) - 2.0
    assert t["net"] == pytest.approx(expected) and t["commission"] == pytest.approx(2.0)
    assert t["cost_pct"] == pytest.approx(-t["net_pct"])                   # flat price: the whole loss is cost


def test_no_look_ahead():
    n = 60
    rng = np.random.default_rng(3)
    a = day_bars(100 + np.cumsum(rng.normal(0, .1, n)))
    b = day_bars(50 + np.cumsum(rng.normal(0, .1, n)))
    e = edges_at(a, 10, A=0.02, B=0.01)
    base = SlotTrader(capital=1000, cost=FREE, hold_minutes=10).run({"A": a, "B": b}, e).trades.iloc[0]
    wrecked = a.copy(); wrecked.iloc[11:] *= 3                              # prices from the entry bar on
    same = SlotTrader(capital=1000, cost=FREE, hold_minutes=10).run({"A": wrecked, "B": b}, e).trades.iloc[0]
    assert same["decision"] == base["decision"] and same["ticker"] == base["ticker"]
    late = a.copy(); late.iloc[40:] *= 3                                     # after the exit: nothing changes
    assert SlotTrader(capital=1000, cost=FREE, hold_minutes=10).run({"A": late, "B": b}, e).trades.iloc[0]["net"] == pytest.approx(base["net"])


def test_summary_and_validation():
    a = day_bars([100.0] * 20 + [102.0] * 40)
    res = SlotTrader(capital=1000, cost=FREE, hold_minutes=30).run({"A": a}, edges_at(a, 5, A=0.05))
    s = res.summary()
    assert s["trades"] == 1 and s["win_rate"] == 1.0 and s["total_return"] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        SlotTrader(capital=0)
    with pytest.raises(ValueError):
        SlotTrader(slots=0)
    with pytest.raises(ValueError):
        SlotTrader(hold_minutes=0)


def test_three_slots_buy_the_three_best_with_a_third_of_the_account_each():
    bars = {k: day_bars([p] * 60) for k, p in (("A", 100.0), ("B", 50.0), ("C", 20.0), ("D", 10.0))}
    e = edges_at(bars["A"], 5, A=0.05, B=0.04, C=0.03, D=0.01)
    res = SlotTrader(capital=3000, slots=3, cost=FREE, hold_minutes=10).run(bars, e)
    assert res.trades["ticker"].tolist() == ["A", "B", "C"]
    assert res.trades["shares"].tolist() == [10, 20, 50]                  # $1,000 each


def test_slots_limit_the_positions_open_at_once_and_the_trades_a_day():
    bars = {k: day_bars([100.0] * 80) for k in "ABCD"}
    idx = pd.DatetimeIndex([bars["A"].index[5], bars["A"].index[10], bars["A"].index[70]])
    nan = np.nan
    e = pd.DataFrame({"A": [0.05, nan, nan], "B": [nan, 0.05, nan], "C": [nan, nan, 0.05]}, index=idx)
    one_busy = SlotTrader(capital=2000, slots=2, cost=FREE, hold_minutes=60, max_trades_per_day=3).run(bars, e)
    assert one_busy.trades["ticker"].tolist() == ["A", "B", "C"]          # A and B overlap in the two slots, C fits after both are sold
    full = SlotTrader(capital=2000, slots=1, cost=FREE, hold_minutes=60, max_trades_per_day=3).run(bars, e)
    assert full.trades["ticker"].tolist() == ["A", "C"]                   # one slot: B blocked while A is held
    assert len(SlotTrader(capital=2000, slots=2, cost=FREE, hold_minutes=5).run(bars, e).trades) == 2   # default: one per slot


def test_the_hurdle_uses_the_slot_value_so_small_slots_need_bigger_edges():
    a = day_bars([100.0] * 60)
    cost = CostModel(spread_bps=2.0)                                       # $1 minimum: 0.24% on $1,000, 0.44% on $500
    edge = edges_at(a, 5, A=0.003)
    assert len(SlotTrader(capital=1000, slots=1, cost=cost).run({"A": a}, edge).trades) == 1
    assert SlotTrader(capital=1000, slots=2, cost=cost).run({"A": a}, edge).trades.empty


def test_a_stock_is_bought_at_most_once_a_day():
    a = day_bars([100.0] * 80)
    idx = pd.DatetimeIndex([a.index[5], a.index[40]])
    res = SlotTrader(capital=2000, slots=2, cost=FREE, hold_minutes=10).run({"A": a}, pd.DataFrame({"A": [0.05, 0.05]}, index=idx))
    assert len(res.trades) == 1


def test_price_cost_and_invested_filters_skip_stocks_even_with_a_big_edge():
    bars = {k: day_bars([p] * 60) for k, p in (("CHEAP", 4.0), ("MID", 60.0), ("PRICEY", 400.0))}
    e = edges_at(bars["MID"], 5, CHEAP=0.2, PRICEY=0.19, MID=0.18)
    cost = ibkr_tiered(spread_bps=2.0)
    base = dict(capital=770, cost=cost, hold_minutes=10)
    assert SlotTrader(**base).run(bars, e).trades["ticker"].iloc[0] == "CHEAP"          # highest edge wins
    assert SlotTrader(**base, min_price=10.0).run(bars, e).trades["ticker"].iloc[0] == "PRICEY"
    assert SlotTrader(**base, min_price=10.0, max_price=300.0).run(bars, e).trades["ticker"].iloc[0] == "MID"
    # a cost ceiling: $4 shares cost about 0.4% in per-share fees on a $770 slot, MID only about 0.15%
    assert SlotTrader(**base, max_cost=0.0025).run(bars, e).trades["ticker"].iloc[0] == "PRICEY"
    # one $400 share covers 52% of the slot: needs min_invested above that to be rejected
    assert SlotTrader(**base, min_price=10.0, min_invested=0.8).run(bars, e).trades["ticker"].iloc[0] == "MID"
    assert SlotTrader(**base, min_price=10.0, min_invested=0.99).run(bars, e).trades.empty


def test_filter_validation():
    for bad in (dict(min_price=-1), dict(min_price=5, max_price=5), dict(min_invested=1.5), dict(max_cost=0)):
        with pytest.raises(ValueError):
            SlotTrader(**bad)
