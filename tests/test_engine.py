import pandas as pd
import pytest

from algotrader import Backtester, CostModel, PositionSizer, SmaCrossStrategy
from helpers import make_bars

STRAT = SmaCrossStrategy(fast=2, slow=3, atr_period=2, atr_mult=2.0)
NO_COST = CostModel(commission_per_share=0, min_commission=0)
BIG = PositionSizer(risk_per_trade=1.0, max_position_frac=0.5)   # sizes by the 50% cap


def engine(costs=NO_COST, sizer=BIG, capital=10_000.0):
    return Backtester(STRAT, costs, sizer, capital)


UP_THEN_DOWN = [10] * 5 + [11, 12, 13, 14, 15] + [14.5, 14, 13.5, 13, 12.5, 12]   # gentle decline: signal exit, no stop


def test_no_trades_in_flat_market():
    res = engine().run(make_bars([10] * 30))
    assert res.trades == []
    assert (res.equity == 10_000).all()


def test_entry_fills_at_next_bar_open_not_signal_close():
    bars = make_bars(UP_THEN_DOWN)
    trade = engine().run(bars).trades[0]
    ind = STRAT.indicators(bars)
    signal_bar = ind.index[ind["want_long"]][0]
    entry_bar = bars.index[bars.index.get_loc(signal_bar) + 1]
    assert trade.entry_time == entry_bar
    assert trade.entry_price == bars.loc[entry_bar, "open"]


def test_exit_on_signal_fills_at_next_open():
    bars = make_bars(UP_THEN_DOWN)
    trade = engine().run(bars).trades[0]
    assert trade.exit_reason == "signal"
    assert trade.exit_price == bars.loc[trade.exit_time, "open"]
    assert trade.exit_time > trade.entry_time


def test_no_look_ahead_prefix_of_equity_is_unchanged_by_future_data():
    bars = make_bars(UP_THEN_DOWN)
    full = engine().run(bars).equity
    cut = engine().run(bars.iloc[:12]).equity
    pd.testing.assert_series_equal(full.iloc[:11], cut.iloc[:11])   # last bar differs (forced close)


def test_position_closed_at_end_of_data():
    res = engine().run(make_bars([10] * 5 + [11, 12, 13, 14, 15]))
    assert res.trades[-1].exit_reason == "end"
    assert res.trades[-1].exit_price == 15


def test_intrabar_stop_fills_at_stop_price():
    closes = [10] * 5 + [11, 12, 13, 14]
    bars = make_bars(closes)
    # Next bar after entry trades down to a low of 1 but closes at 13 (stop is well above).
    extra = make_bars([13], opens=[14], start=bars.index[-1] + pd.Timedelta("1min"))
    extra["low"] = 1.0
    res = engine().run(pd.concat([bars, extra]))
    stop_trades = [t for t in res.trades if t.exit_reason == "stop"]
    assert len(stop_trades) == 1
    t = stop_trades[0]
    atr_at_signal = STRAT.indicators(bars)["atr"].loc[bars.index[
        bars.index.get_loc(t.entry_time) - 1]]
    assert t.exit_price == pytest.approx(t.entry_price - 2.0 * atr_at_signal)


def test_gap_below_stop_fills_at_open():
    bars = make_bars([10] * 5 + [11, 12, 13, 14], pad=0.5)
    gap = make_bars([3], opens=[3], start=bars.index[-1] + pd.Timedelta("1min"))
    res = engine().run(pd.concat([bars, gap]))
    t = [t for t in res.trades if t.exit_reason == "stop"][0]
    assert t.exit_price == 3.0          # open, which is below the stop


def test_no_reentry_after_stop_until_new_crossover():
    bars = make_bars([10] * 5 + [11, 12, 13, 14])
    crash = make_bars([14] * 6, opens=[2] + [14] * 5,
                      start=bars.index[-1] + pd.Timedelta("1min"))
    res = engine().run(pd.concat([bars, crash]))
    assert [t.exit_reason for t in res.trades] == ["stop"]


def test_commission_charged_on_both_sides_and_reduces_equity():
    costs = CostModel(commission_per_share=0, min_commission=1.0)
    bars = make_bars(UP_THEN_DOWN)
    free = engine().run(bars)
    paid = engine(costs=costs).run(bars)
    assert paid.trades[0].commission == pytest.approx(2.0)
    assert paid.equity.iloc[-1] == pytest.approx(free.equity.iloc[-1] - 2.0)


def test_slippage_worsens_fills():
    bars = make_bars(UP_THEN_DOWN)
    base = engine().run(bars).trades[0]
    slipped = engine(costs=CostModel(0, 0, slippage_per_share=0.1)).run(bars).trades[0]
    assert slipped.entry_price == pytest.approx(base.entry_price + 0.1)
    assert slipped.exit_price == pytest.approx(base.exit_price - 0.1)


def test_accounting_identity_final_equity_equals_capital_plus_trade_pnl():
    costs = CostModel(commission_per_share=0.005, min_commission=1.0, slippage_per_share=0.01)
    res = engine(costs=costs).run(make_bars(UP_THEN_DOWN * 3))
    assert res.trades
    assert res.equity.iloc[-1] == pytest.approx(
        res.initial_capital + sum(t.pnl for t in res.trades))


def test_position_size_never_exceeds_cap_and_cash_never_negative():
    res = engine(sizer=PositionSizer(1.0, 0.5)).run(make_bars(UP_THEN_DOWN * 3))
    prior_equity = res.equity.shift().bfill()      # equity at the previous close
    for t in res.trades:
        assert t.shares * t.entry_price <= 0.5 * prior_equity[t.entry_time] + 1e-6
    assert (res.equity > 0).all()


def test_trades_frame_columns_and_empty_case():
    empty = engine().run(make_bars([10] * 10)).trades_frame()
    assert empty.empty and "pnl" in empty.columns
    full = engine().run(make_bars(UP_THEN_DOWN)).trades_frame()
    assert len(full) == 1 and full.loc[0, "pnl"] != 0


def test_position_series_matches_trades():
    bars = make_bars(UP_THEN_DOWN)
    res = engine().run(bars)
    t = res.trades[0]
    held = res.position[res.position > 0]
    assert (held == t.shares).all()
    assert held.index[0] == t.entry_time
    assert held.index[-1] < t.exit_time               # flat at the close of the exit bar
    assert res.position.iloc[-1] == 0


def test_end_of_data_close_leaves_position_flat():
    res = engine().run(make_bars([10] * 5 + [11, 12, 13, 14, 15]))
    assert res.position.iloc[-1] == 0 and res.position.iloc[-2] > 0


def test_empty_bars_rejected():
    with pytest.raises(ValueError):
        engine().run(make_bars([10]).iloc[:0])


class AlwaysLong:
    """Minimal custom strategy: long from bar 2 on, fixed ATR. Shows the engine
    accepts anything with `indicators` and `atr_mult` (the Strategy protocol)."""
    atr_mult = 1.0

    def indicators(self, bars):
        df = bars.copy()
        df["atr"] = 1.0
        df["want_long"] = [i >= 2 for i in range(len(df))]
        return df


def test_engine_accepts_any_strategy_with_the_protocol():
    bars = make_bars([10, 10, 10, 11, 12])
    res = Backtester(AlwaysLong(), NO_COST, BIG).run(bars)
    assert len(res.trades) == 1
    assert res.trades[0].entry_time == bars.index[3]
    assert res.trades[0].exit_reason == "end"


class Scripted:
    """Strategy whose columns are given directly, for testing optional engine columns."""
    atr_mult = 1.0

    def __init__(self, want, stop_price=None, exit_at_close=None):
        self.want, self.stop_price, self.exit_at_close = want, stop_price, exit_at_close

    def indicators(self, bars):
        df = bars.copy()
        df["atr"] = 1.0
        df["want_long"] = self.want
        if self.stop_price is not None:
            df["stop_price"] = self.stop_price
        if self.exit_at_close is not None:
            df["exit_at_close"] = self.exit_at_close
        return df


def test_start_uses_earlier_bars_only_as_warmup():
    bars = make_bars([10, 10, 10, 11, 12, 13, 14])
    res = Backtester(AlwaysLong(), NO_COST, BIG).run(bars, start=bars.index[4])
    assert res.equity.index[0] == bars.index[4] and len(res.equity) == 3
    assert res.trades[0].entry_time == bars.index[5]          # signal at the start bar's close
    with pytest.raises(ValueError):
        Backtester(AlwaysLong(), NO_COST, BIG).run(bars, start="2030-01-01")


def test_explicit_stop_price_replaces_atr_stop_and_sets_size():
    bars = make_bars([10, 10, 10, 10, 10], pad=0.5)            # lows at 9.5
    want = [False, True, True, True, True]
    tight = Backtester(Scripted(want, stop_price=9.8), NO_COST,
                       PositionSizer(risk_per_trade=0.01, max_position_frac=1.0)).run(bars)
    trade = tight.trades[0]
    assert trade.exit_reason == "stop" and trade.exit_price == pytest.approx(9.8)
    assert trade.shares == 500                                   # 1% of 10,000 / (10 - 9.8)
    loose = Backtester(Scripted(want, stop_price=5.0), NO_COST, BIG).run(bars)
    assert loose.trades[0].exit_reason == "end"


def test_no_entry_when_fill_is_below_stop_price():
    bars = make_bars([10, 10, 10, 10])
    res = Backtester(Scripted([False, True, True, True], stop_price=11.0), NO_COST, BIG).run(bars)
    assert res.trades == []


def test_exit_at_close_flattens_and_blocks_new_orders():
    bars = make_bars([10, 11, 12, 13, 14, 15])
    want = [False, True, True, False, True, True]
    eod = [False, False, True, False, False, False]
    res = Backtester(Scripted(want, exit_at_close=eod), NO_COST, BIG).run(bars)
    first = res.trades[0]
    assert first.exit_reason == "close" and first.exit_time == bars.index[2]
    assert first.exit_price == 12                                # that bar's close
    assert res.position.iloc[2] == 0
    # want stays True on the exit bar but no order is placed from it; the next entry
    # needs a fresh signal (bar 4) and fills at bar 5's open
    assert res.trades[1].entry_time == bars.index[5]


def test_profit_target_in_r_multiples():
    import numpy as np
    import pandas as pd
    from algotrader import Backtester, CostModel, PositionSizer

    class TargetStrat:
        atr_mult = 50.0

        def indicators(self, bars):
            df = bars.copy()
            df["want_long"] = [False, True, True, True, True]
            df["atr"] = 1.0
            df["stop_price"] = 99.0
            df["target_r"] = 1.0
            return df

    idx = pd.date_range("2024-01-02 09:30", periods=5, freq="min")
    bars = pd.DataFrame({"open": [100, 100, 100.5, 101, 101], "high": [100, 100.5, 102.5, 103, 103],
                         "low": [100, 99.8, 100.2, 100.8, 100.8], "close": [100, 100.2, 102, 102, 102],
                         "volume": 1}, index=idx, dtype=float)
    res = Backtester(TargetStrat(), CostModel(0, 0, max_commission_frac=0), PositionSizer(1.0, 1.0)).run(bars)
    [t] = res.trades
    assert t.entry_price == 100.5 and t.exit_reason == "target"
    assert t.exit_price == pytest.approx(100.5 + 1.0 * (100.5 - 99.0))   # 102.0, high 102.5
    assert t.exit_time == idx[2]
