import numpy as np
import pandas as pd
import pytest

from algotrader.minutefeat import (DECISIONS, FEATURES, HORIZONS, LABELS, early_close_days, membership_by_day,
                                   month_tickers, process)

DAYS = [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")]


def bars(ticker, day, minutes, base=100.0, slope=0.1, volume=100.0):
    t = day + pd.Timedelta(hours=9, minutes=30) + pd.to_timedelta(list(minutes), unit="min")
    o = base + slope * np.array(list(minutes), dtype=float)
    return pd.DataFrame({"ticker": ticker, "t": t, "open": o, "high": o + 0.07, "low": o - 0.02, "close": o + 0.05,
                         "volume": volume})


def frame(tickers=("AAA", "SPY"), days=DAYS, drop=None, **kw):
    parts = []
    for d in days:
        for k in tickers:
            ms = [m for m in range(390) if not (drop and (k, d, m) in drop)]
            parts.append(bars(k, d, ms, base=kw.get("base", 100.0) if k != "SPY" else 400.0))
    return pd.concat(parts, ignore_index=True)


def members(*tickers, days=DAYS):
    return pd.DataFrame({t: True for t in tickers}, index=pd.DatetimeIndex(days))


def row(res, day, ticker, minute):
    r = res[(res["date"] == day) & (res["ticker"] == ticker) & (res["minute"] == minute)]
    assert len(r) == 1, (day, ticker, minute, len(r))
    return r.iloc[0]


def test_one_row_per_member_day_and_decision_with_all_columns():
    res, _ = process(frame(), members("AAA"))
    assert set(res["ticker"]) == {"AAA"} and len(res) == 2 * len(DECISIONS)
    assert list(res.columns) == ["date", "ticker"] + FEATURES + LABELS


def test_forward_returns_follow_the_slot_trader_exit_rule():
    res, _ = process(frame(), members("AAA"))
    r = row(res, DAYS[0], "AAA", 30)                                   # decide at 10:00, buy the open of minute 30
    o = lambda m: 100 + 0.1 * m
    for h in HORIZONS:
        assert r[f"fwd{h}"] == pytest.approx(o(30 + h) / o(30) - 1, rel=1e-4)
    assert r["fwd_close"] == pytest.approx((o(389) + 0.05) / o(30) - 1, rel=1e-4)
    late = row(res, DAYS[0], "AAA", 330)                               # 15:00: 240 minutes do not fit: the last close
    assert late["fwd240"] == pytest.approx(late["fwd_close"])
    assert late["fwd30"] == pytest.approx(o(360) / o(330) - 1, rel=1e-4)


def test_missing_minutes_exit_at_the_next_bar_and_no_entry_without_a_trade():
    drop = {("AAA", DAYS[0], 40), ("AAA", DAYS[0], 41), ("AAA", DAYS[0], 60)}
    res, _ = process(frame(drop=drop), members("AAA"))
    r = row(res, DAYS[0], "AAA", 30)
    assert r["fwd10"] == pytest.approx((100 + 0.1 * 42) / (100 + 0.1 * 30) - 1, rel=1e-4)      # first bar at or after 40
    assert not ((res["date"] == DAYS[0]) & (res["minute"] == 60)).any()                         # no trade in the entry minute


def test_features_hand_calculation():
    res, _ = process(frame(), members("AAA"))
    r = row(res, DAYS[0], "AAA", 60)                                   # decision at the end of minute 59
    c = lambda m: 100 + 0.1 * m + 0.05
    assert r["ret5"] == pytest.approx(c(59) / c(54) - 1, rel=1e-4)
    assert r["ret30"] == pytest.approx(c(59) / c(29) - 1, rel=1e-4)
    assert np.isnan(r["ret60"])                                          # not 60 minutes of history yet
    r2 = row(res, DAYS[0], "AAA", 120)
    assert r2["ret60"] == pytest.approx(c(119) / c(59) - 1, rel=1e-4)
    assert r["ret_open"] == pytest.approx(c(59) / 100.0 - 1, rel=1e-4)
    tp = np.array([(100 + 0.1 * m + 0.07 + 100 + 0.1 * m - 0.02 + 100 + 0.1 * m + 0.05) / 3 for m in range(60)])
    assert r["vwap_dist"] == pytest.approx(c(59) / tp.mean() - 1, rel=1e-4)
    assert r["range_pos"] == pytest.approx((c(59) - 99.98) / (100 + 5.9 + 0.07 - 99.98), rel=1e-4)
    assert r["active30"] == 30 and r["minute"] == 60 and r["mins_left"] == 330 and r["price"] == pytest.approx(c(59), rel=1e-5)
    assert r["range_bp"] == pytest.approx(0.09 / c(59) * 1e4, rel=2e-2)
    assert r["spy_ret30"] == pytest.approx((400 + 5.9 + 0.05) / (400 + 2.9 + 0.05) - 1, rel=1e-4)
    assert r["rel30"] == pytest.approx(r["ret30"] - r["spy_ret30"], rel=1e-4)


def test_previous_session_gap_volume_and_volatility_come_from_the_carry():
    old = pd.bdate_range("2024-02-22", periods=7)
    carry = pd.DataFrame({"date": list(old) * 2, "ticker": ["AAA"] * 7 + ["SPY"] * 7,
                          "close": [95.0] * 7 + [390.0] * 7, "volume": [39000.0] * 14})
    res, out = process(frame(), members("AAA"), carry)
    r = row(res, DAYS[0], "AAA", 30)
    assert r["gap"] == pytest.approx(100.0 / 95.0 - 1, rel=1e-5)
    assert r["rvol_day"] == pytest.approx(3000.0 / 39000.0, rel=1e-4)      # 30 bars of 100 shares
    assert r["dvol20"] == pytest.approx(0.0, abs=1e-9) and r["prev_ret"] == pytest.approx(0.0, abs=1e-9)
    assert r["logdv20"] == pytest.approx(np.log(39000.0 * 95.0), rel=1e-6)
    nxt = row(res, DAYS[1], "AAA", 30)
    assert nxt["gap"] == pytest.approx(100.0 / (100 + 38.9 + 0.05) - 1, rel=1e-4)
    assert nxt["prev_ret"] == pytest.approx((100 + 38.9 + 0.05) / 95.0 - 1, rel=1e-4)
    assert set(out["ticker"]) == {"AAA", "SPY"} and out["date"].max() == DAYS[1]
    res2, _ = process(frame(), members("AAA"))                              # no carry: no previous close
    assert np.isnan(row(res2, DAYS[0], "AAA", 30)["gap"])


def test_early_close_sessions_are_cut_at_one_oclock():
    d1 = DAYS[1]
    spy = bars("SPY", d1, range(210))
    spy = pd.concat([spy, bars("SPY", d1, [215, 240, 300, 340], base=400.0)])                      # after-hours prints
    aaa = bars("AAA", d1, list(range(210)) + list(range(215, 390)))                              # trades after 13:00 are not regular
    day0 = frame(days=[DAYS[0]])
    both = pd.concat([day0, spy, aaa], ignore_index=True)
    assert early_close_days(both[both["ticker"] == "SPY"].assign(t=both["t"])) == {d1}
    res, _ = process(both, members("AAA"))
    assert res[res["date"] == d1]["minute"].max() == 180                                           # 12:30 is the last decision
    r = row(res, d1, "AAA", 180)
    assert r["mins_left"] == 30
    assert r["fwd_close"] == pytest.approx((100 + 0.1 * 209 + 0.05) / (100 + 0.1 * 180) - 1, rel=1e-4)
    assert r["fwd90"] == pytest.approx(r["fwd_close"])
    assert row(res, DAYS[0], "AAA", 330)["mins_left"] == 60


def test_only_members_get_rows_and_membership_is_by_day():
    df = frame(tickers=("AAA", "BBB", "SPY"))
    mem = pd.DataFrame({"AAA": [True, True], "BBB": [False, True]}, index=pd.DatetimeIndex(DAYS))
    res, _ = process(df, mem)
    assert set(res[res["date"] == DAYS[0]]["ticker"]) == {"AAA"}
    assert set(res[res["date"] == DAYS[1]]["ticker"]) == {"AAA", "BBB"}
    long = pd.DataFrame({"date": [DAYS[0], DAYS[0], DAYS[1]], "ticker": ["AAA", "BBB", "AAA"]})
    m = membership_by_day(long, pd.DatetimeIndex(DAYS), ["AAA", "BBB"])
    assert m.loc[DAYS[1]].tolist() == [True, False] or m.loc[DAYS[1]].tolist() == [True, True]


def test_features_do_not_see_later_bars():
    df = frame()
    base, _ = process(df, members("AAA"))
    wrecked = df.copy()
    late = (wrecked["ticker"] == "AAA") & (wrecked["t"] >= DAYS[0] + pd.Timedelta(hours=10, minutes=35))     # from minute 65 on
    wrecked.loc[late, ["open", "high", "low", "close"]] *= 3
    wrecked.loc[late, "volume"] *= 50
    w, _ = process(wrecked, members("AAA"))
    a, b = row(base, DAYS[0], "AAA", 60), row(w, DAYS[0], "AAA", 60)
    for f in FEATURES:
        assert a[f] == pytest.approx(b[f], rel=1e-6, nan_ok=True), f
    assert a["fwd10"] != pytest.approx(b["fwd10"])                                                  # labels do look ahead, by design


def test_month_tickers_use_the_list_in_force_across_gaps_and_after_the_last_list():
    long = pd.DataFrame({"date": [pd.Timestamp("2022-06-28")] * 2 + [pd.Timestamp("2022-09-19")] * 3,
                         "ticker": ["A", "B", "A", "C", "D"]})
    month = lambda m: month_tickers(long, pd.Period(m, "M").start_time, pd.Period(m, "M").end_time)
    assert month("2022-08") == {"A", "B"}                       # no list inside the month: the one in force
    assert month("2022-09") == {"A", "B", "C", "D"}             # the old list at the start and the new one
    assert month("2023-03") == {"A", "C", "D"}                  # after the last list: carried forward
