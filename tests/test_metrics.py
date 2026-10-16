import math

import pandas as pd
import pytest

from algotrader import BacktestResult, Metrics, Trade
from helpers import make_bars


def result(equity, trades=(), capital=100.0, start="2024-01-01", freq="1D"):
    idx = pd.date_range(start, periods=len(equity), freq=freq)
    return BacktestResult(capital, pd.Series(equity, index=idx, dtype=float), list(trades))


def trade(pnl, shares=1):
    t = pd.Timestamp("2024-01-01")
    return Trade(t, t, shares, 10.0, 10.0 + pnl / shares, 0.0, "signal")


def test_total_return_and_drawdown():
    m = Metrics(result([100, 120, 90, 110]))
    assert m.total_return() == pytest.approx(0.10)
    assert m.max_drawdown() == pytest.approx(90 / 120 - 1)


def test_no_drawdown_when_equity_only_rises():
    assert Metrics(result([100, 101, 102])).max_drawdown() == 0


def test_sharpe_nan_for_constant_equity():
    assert math.isnan(Metrics(result([100] * 10)).sharpe())


def test_sharpe_matches_manual_calculation():
    eq = [100, 101, 100, 103, 104]
    r = pd.Series(eq).pct_change().dropna()
    expected = r.mean() / r.std() * math.sqrt(252)
    assert Metrics(result(eq)).sharpe() == pytest.approx(expected)


def test_sharpe_uses_daily_not_per_bar_returns():
    # Two bars a day; only the last equity of each day counts.
    eq = [100, 500, 101, 500, 102, 500]       # day-end values 500 apart from the intraday...
    m = Metrics(result(eq, freq="12h"))
    assert list(m.daily_equity) == [500, 500, 500]


def test_cagr_doubles_in_one_year():
    eq = result([100, 200], start="2023-01-01", freq="365D")
    assert Metrics(eq).cagr() == pytest.approx(1.0, rel=0.01)


def test_cagr_nan_when_wiped_out():
    assert math.isnan(Metrics(result([100, 0])).cagr())


def test_trade_stats():
    stats = Metrics(result([100, 100], [trade(10), trade(-4), trade(6)])).trade_stats()
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_win"] == 8 and stats["avg_loss"] == -4
    assert stats["profit_factor"] == pytest.approx(16 / 4)


def test_profit_factor_infinite_without_losses():
    stats = Metrics(result([100, 100], [trade(5), trade(3)])).trade_stats()
    assert stats["profit_factor"] == math.inf


def test_trade_stats_with_no_trades():
    stats = Metrics(result([100, 100])).trade_stats()
    assert stats["trades"] == 0 and math.isnan(stats["win_rate"])


def test_buy_and_hold_from_first_open_to_last_close():
    bars = make_bars([10, 12, 15], opens=[10, 11, 14])
    assert Metrics(result([100, 100]), bars).buy_and_hold() == pytest.approx(0.5)
    assert math.isnan(Metrics(result([100, 100])).buy_and_hold())


def test_exposure_is_fraction_of_bars_in_market():
    res = result([100] * 4)
    res.position = pd.Series([0, 5, 5, 0], index=res.equity.index)
    assert Metrics(res).exposure() == pytest.approx(0.5)
    assert math.isnan(Metrics(result([100] * 4)).exposure())


def test_summary_has_all_keys():
    keys = Metrics(result([100, 110])).summary().keys()
    assert {"total_return", "cagr", "sharpe", "max_drawdown", "buy_and_hold", "exposure",
            "final_equity", "trades", "win_rate", "profit_factor"} <= set(keys)


def test_buy_and_hold_sharpe_equals_sharpe_of_holding():
    closes = [10, 11, 10.5, 12, 11.8]
    bars = make_bars(closes, opens=[10, 10.8, 10.9, 10.6, 12], freq="1D")
    # equity of holding: start at the first open, then each daily close
    hold = result([10] + closes, capital=10, start="2024-01-01")
    expected = Metrics(hold).sharpe()
    assert Metrics(result([100] * 5), bars).buy_and_hold_sharpe() == pytest.approx(expected)
    assert math.isnan(Metrics(result([100, 100])).buy_and_hold_sharpe())
