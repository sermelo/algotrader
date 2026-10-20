import numpy as np
import pandas as pd
import pytest

from algotrader import mergerarb as ma
from algotrader.costs import CostModel
from algotrader.portfolio import PortfolioBacktester

CAL = pd.bdate_range("2024-01-01", periods=60)


def daily(closes, start=0, volume=1e6):
    idx = CAL[start:start + len(closes)]
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "close": c, "volume": volume}, index=idx)


def filing(day, tickers, cash, cik=1, acc="a1", hour=12, form="SC TO-T"):
    return {"cik": cik, "accession": acc, "form": form, "filed": str(CAL[day].date()),
            "accepted": f"{CAL[day].date()}T{hour:02d}:00:00.000Z".replace("Z", "-05:00"),
            "tickers": tickers, "cash": cash}


# ---------------------------------------------------------------- parsing

def test_tickers_from_exchange_tags_and_symbol_quotes():
    t = ma.clean_text("<p>Acme (NASDAQ: ACME) and Big (NYSE:BIG); shares trade under the symbol "
                      "&#8220;ACM&#8221;.</p>".encode())
    assert ma.tickers(t) == ["ACM", "ACME", "BIG"]
    assert ma.tickers("nothing here, Nasdaq rules apply") == []


def test_cash_price_most_named_and_formats():
    t = ("receive $12.50 in cash per share ... $12.50 per Share, net to the seller in cash ... "
         "a $3.00 in cash fee")
    assert ma.cash_price(t) == 12.5
    assert ma.cash_price("$1,050.25 per share in cash") == 1050.25
    assert np.isnan(ma.cash_price("an exchange ratio of 0.5 shares"))


# ---------------------------------------------------------------- deals

def prices():
    return {"TGT": daily([10.0] * 30 + [11.5] * 30),          # target jumps towards the 12.0 offer
            "ACQ": daily([50.0] * 60),                          # acquirer, far from the offer
            "THIN": daily([11.5] * 60, volume=10)}              # illiquid


def test_deal_needs_exactly_one_ticker_below_offer():
    f = pd.DataFrame([filing(35, "ACQ|TGT", 12.0)])
    d = ma.find_deals(f, prices(), CAL)
    assert list(d["ticker"]) == ["TGT"] and d["day"].iloc[0] == CAL[35]
    assert d["spread"].iloc[0] == pytest.approx(12.0 / 11.5 - 1)
    # above the offer (spread <= 0), too far below it, illiquid: no deal
    for cash, tick in ((11.0, "TGT"), (20.0, "TGT"), (12.0, "THIN")):
        assert ma.find_deals(pd.DataFrame([filing(35, tick, cash)]), prices(), CAL).empty


def test_after_close_filing_decides_next_day_with_its_close():
    f = pd.DataFrame([filing(29, "TGT", 12.0, hour=17)])        # accepted 17:00 on day 29
    d = ma.find_deals(f, prices(), CAL)
    assert d["day"].iloc[0] == CAL[30] and d["close"].iloc[0] == 11.5


def test_cooldown_blank_check_and_missing_fields():
    f = pd.DataFrame([filing(35, "TGT", 12.0, acc="a1"), filing(40, "TGT", 12.0, acc="a2"),
                      filing(41, "", 12.0, acc="a3"), filing(42, "TGT", np.nan, acc="a4")])
    d = ma.find_deals(f, prices(), CAL)
    assert list(d["accession"]) == ["a1"]
    assert ma.find_deals(f, prices(), CAL, blank_check=frozenset({1})).empty


def test_deals_unchanged_when_later_prices_are_wrecked():
    f = pd.DataFrame([filing(35, "ACQ|TGT", 12.0)])
    p = prices()
    p["TGT"].iloc[36:, :2] = 0.01
    pd.testing.assert_frame_equal(ma.find_deals(f, p, CAL), ma.find_deals(f, prices(), CAL))


def test_termination_ignores_completion_and_earlier_filings():
    deals = pd.DataFrame({"cik": [1], "day": [CAL[35]]})
    e = pd.DataFrame({"cik": [1, 1, 1, 2],
                      "accepted": [f"{CAL[i].date()}T12:00:00.000-05:00" for i in (30, 40, 45, 38)],
                      "filed": [str(CAL[i].date()) for i in (30, 40, 45, 38)],
                      "items": ["1.02", "1.02,2.01,3.01", "1.02,9.01", "1.02"]})
    assert ma.termination_days(deals, e, CAL).iloc[0] == CAL[45]


# ---------------------------------------------------------------- portfolio

def deals_frame(rows):
    d = pd.DataFrame(rows, columns=["ticker", "day", "termination"])
    d["name_"] = ma.deal_names(d)
    return d


def test_schedule_slots_and_exit():
    frames = {"A@20240102": daily([1] * 20), "B@20240103": daily([1] * 40), "C@20240104": daily([1] * 40)}
    d = deals_frame([("A", CAL[1], pd.NaT), ("B", CAL[2], CAL[10]), ("C", CAL[3], pd.NaT)])
    held = ma.schedule(d, frames, slots=2)
    assert list(held["name"]) == ["A@20240102", "B@20240103"]           # C: both slots busy
    assert held["exit"].tolist() == [CAL[19], CAL[10]]                 # last bar, termination
    held = ma.schedule(d, frames, slots=3, max_days=7)
    assert held["exit"].iloc[2] == CAL[3] + pd.Timedelta(days=7)


def test_engine_buys_next_open_and_cashes_out_at_last_close():
    tgt = daily([10.0] * 5 + [11.5] * 10)                             # deal closes after day 14
    tgt.loc[CAL[6], "open"] = 11.6
    cal = daily([100.0] * 30)
    held = pd.DataFrame({"name": ["T"], "day": [CAL[5]], "exit": [CAL[14]]})
    res = PortfolioBacktester(ma.MergerArb(held, slots=2), CostModel(0, 0, 0, 0), 10_000.0).run(
        {"T": tgt, "SPY": cal})
    o = res.orders
    assert o[0][0] == CAL[6] and o[0][2] == int(5000 / 11.6) and o[0][3] == 11.6
    assert o[-1][0] == CAL[14] and o[-1][3] == 11.5
    d = ma.deal_returns(res, held)
    assert d["return"].iloc[0] == pytest.approx(11.5 / 11.6 - 1)
    closes = pd.DataFrame({"T": tgt["close"], "SPY": cal["close"]})
    rf = pd.Series(0.001, index=CAL[:30])
    r = ma.with_cash_rate(res, closes, rf)
    assert r.loc[CAL[3]] == pytest.approx(0.001)                      # all cash
    assert r.loc[CAL[20]] == pytest.approx(0.001)                     # cash again after the deal
    held_share = 1 - (res.holdings["T"] * closes["T"]).loc[CAL[6]] / res.equity.loc[CAL[6]]
    assert r.loc[CAL[7]] == pytest.approx(0.001 * held_share)          # price flat: only interest


def test_runner_experiments_are_counted_once_each():
    assert list(ma.SLOTS) == ["merger_arb", "merger_arb_10", "merger_arb_40"]


def test_summary_applies_the_preregistered_rule(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(0.0004, index=idx)
    row = {"experiment": "merger_arb", "base_cagr": 0.1, "base_excess_yr": 0.05, "base_sharpe_excess": 1.0,
           "sharpe_lo": 0.5, "sharpe_hi": 1.5, "ff_alpha": 0.05, "ff_alpha_t": 2.1, "ff_beta": 0.1,
           "max_drawdown": -0.05, "gross_excess_yr": 0.06, "stress_excess_yr": 0.01, "deals": 10,
           "deal_mean": 0.02, "deal_median": 0.01, "deals_lost_10pct": 0.1, "turnover": 3.0}
    deals = pd.DataFrame({"day": [idx[0], idx[-1]]})
    res = {"base": r, "spy": r}
    assert "Verdict: pass" in ma.write_summary(tmp_path, [row], res, deals)
    assert "Verdict: fail" in ma.write_summary(tmp_path, [row | {"ff_alpha_t": 1.9}], res, deals)
    assert "Verdict: fail" in ma.write_summary(tmp_path, [row | {"stress_excess_yr": -0.01}], res, deals)
