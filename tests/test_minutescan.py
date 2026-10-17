import numpy as np
import pandas as pd
import pytest

from algotrader.minutefeat import LABELS
from algotrader.minutescan import BinTable, SignalScan, quantile_edges

DATES = pd.bdate_range("2020-01-06", periods=40)


def dataset(seed=0, n_stocks=60, signal=0.002, vol_link=True):
    """Stocks x decisions; `f_good` predicts fwd10, `f_noise` does not, `vol` scales the size of moves."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in DATES:
        for minute in (30, 60):
            good = rng.normal(size=n_stocks)
            noise = rng.normal(size=n_stocks)
            vol = rng.uniform(0.5, 3.0, n_stocks)
            market = rng.normal(0, 0.001)
            ret = market + signal * good + 0.003 * vol * rng.normal(size=n_stocks) * (1 if vol_link else 0.5)
            df = pd.DataFrame({"date": d, "minute": float(minute), "ticker": [f"S{i}" for i in range(n_stocks)],
                               "f_good": good, "f_noise": noise, "vol": vol})
            for lab in LABELS:
                df[lab] = ret
            rows.append(df)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "minute", "ticker"], ignore_index=True)


def test_signal_scan_finds_the_planted_feature_and_not_the_noise():
    scan = SignalScan(features=["f_good", "f_noise"], labels=["fwd10"])
    scan.add(dataset())
    r = scan.result().set_index("feature")
    assert r.loc["f_good", "ic"] > 0.2 and r.loc["f_good", "ic_t"] > 5
    assert abs(r.loc["f_noise", "ic_t"]) < 3
    assert r.loc["f_good", "years_positive"] == r.loc["f_good", "years"] == 1
    assert r.loc["f_good", "top10_bp"] > 0 > r.loc["f_good", "bot10_bp"]            # excess return of the top and bottom tenth
    assert r.loc["f_good", "top1_bp"] > r.loc["f_good", "bot1_bp"]
    assert r.loc["f_good", "days"] == len(DATES)


def test_excess_returns_remove_the_market_move_and_batches_add_up():
    df = dataset(seed=1)
    one = SignalScan(features=["f_good"], labels=["fwd10"]); one.add(df)
    split = SignalScan(features=["f_good"], labels=["fwd10"])
    half = DATES[20]
    split.add(df[df["date"] < half]); split.add(df[df["date"] >= half])
    a, b = one.result().iloc[0], split.result().iloc[0]
    assert a["ic"] == pytest.approx(b["ic"]) and a["top1_bp"] == pytest.approx(b["top1_bp"])
    shifted = df.assign(fwd10=df["fwd10"] + 0.05)                                  # a market-wide shift changes nothing
    s = SignalScan(features=["f_good"], labels=["fwd10"]); s.add(shifted)
    assert s.result().iloc[0]["top10_bp"] == pytest.approx(a["top10_bp"], abs=1e-6)
    assert s.result().iloc[0]["top1raw_bp"] == pytest.approx(a["top1raw_bp"] + 500, abs=1e-6)


def test_signal_scan_handles_missing_feature_values():
    df = dataset(seed=2)
    df.loc[df["minute"] == 30.0, "f_good"] = np.nan
    s = SignalScan(features=["f_good"], labels=["fwd10"]); s.add(df)
    assert s.result().iloc[0]["ic_t"] > 3


def test_bintable_shows_bigger_moves_where_volatility_is_higher_and_counts_thresholds():
    from algotrader import minutescan
    minutescan.CONDITIONS["vol"] = lambda d: d["vol"]
    df = dataset(seed=3)
    edges = quantile_edges(df["vol"], 4)
    t = BinTable("vol", edges, labels=["fwd10"], thresholds=(0.001, 0.004))
    t.add(df)
    tab = t.table()
    assert len(tab) == 4 and tab["n"].sum() == len(df)
    assert list(tab["mean_abs"]) == sorted(tab["mean_abs"])                         # monotone in volatility
    assert tab["either_0.001"].iloc[-1] > tab["either_0.001"].iloc[0]
    assert (tab["either_0.004"] <= tab["either_0.001"]).all() and (tab["up_0.001"] <= tab["either_0.001"]).all()
    assert tab["bin_value"].is_monotonic_increasing
    del minutescan.CONDITIONS["vol"]


def test_bintable_discrete_condition_by_minute_and_quantile_edges():
    df = dataset(seed=4)
    t = BinTable("minute", labels=["fwd10"])
    t.add(df)
    tab = t.table()
    assert list(tab["bin_value"]) == [30.0, 60.0] and tab["n"].sum() == len(df)
    t.add(df)
    assert t.table()["n"].sum() == 2 * len(df)
    e = quantile_edges(np.arange(1001.0), 10)
    assert len(e) == 9 and e[4] == pytest.approx(500.0)


def test_rule_scan_picks_the_planted_signal_and_nets_the_cost():
    from algotrader.minutescan import RuleScan
    df = dataset(seed=5, signal=0.004)
    rs = RuleScan(["f_good", "f_noise"], {"all": lambda d: d["vol"] > 0, "volatile": lambda d: d["vol"] >= 2.0}, ks=(1, 3),
                  labels=["fwd10"], costs={1: 0.0008, 3: 0.0014})
    rs.add(df[df["date"] < DATES[20]]); rs.add(df[df["date"] >= DATES[20]])
    r = rs.result()
    pick = lambda **kw: r[(r["filter"] == kw["f"]) & (r["score"] == kw["s"]) & (r["pick"] == kw["p"]) & (r["k"] == kw["k"]) & (r["minute"] == 30)].iloc[0]
    hi, lo, nz = pick(f="all", s="f_good", p="high", k=1), pick(f="all", s="f_good", p="low", k=1), pick(f="all", s="f_noise", p="high", k=1)
    assert hi["excess_bp"] > 30 and hi["excess_t"] > 5 and lo["excess_bp"] < -30        # picks the best / worst by the planted signal
    assert abs(nz["excess_t"]) < 3.5
    assert hi["net_bp"] == pytest.approx(hi["gross_bp"] - 8.0)                          # one slot: 0.08% round trip
    assert pick(f="all", s="f_good", p="high", k=3)["net_bp"] == pytest.approx(pick(f="all", s="f_good", p="high", k=3)["gross_bp"] - 14.0)
    assert pick(f="all", s="f_good", p="high", k=3)["excess_bp"] < hi["excess_bp"]      # three picks dilute the best one
    v = pick(f="volatile", s="f_good", p="high", k=1)
    assert v["days"] == len(DATES) and v["filter_mean_bp"] != pytest.approx(hi["filter_mean_bp"])   # a different filtered set
    assert 0 <= hi["win_net"] <= 1 and hi["years"] == 1


def test_derived_z_scores_scale_moves_by_the_stocks_own_volatility():
    from algotrader.minutescan import derived
    df = pd.DataFrame({"vol30": [4.0, 8.0, 0.0], "dvol20": [0.02, 0.01, 0.0], "ret5": [0.002, 0.002, 0.001], "ret30": [0.006, 0.006, 0.001],
                       "ret_open": [0.01, 0.01, 0.0], "gap": [0.01, 0.01, 0.0], "minute": [390.0, 90.0, 30.0]})
    d = derived(df)
    assert d["z5"].iloc[0] == pytest.approx(0.002 / (4e-4 * np.sqrt(5)))
    assert d["z5"].iloc[0] == pytest.approx(2 * d["z5"].iloc[1])               # same move, half the volatility: twice the z
    assert d["z30"].iloc[1] == pytest.approx(0.006 / (8e-4 * np.sqrt(30)))
    assert d["z_open"].iloc[1] == pytest.approx(0.01 / (0.01 * np.sqrt(90 / 390)))
    assert d["z_gap"].iloc[0] == pytest.approx(0.5) and np.isnan(d["z5"].iloc[2]) and np.isnan(d["z_gap"].iloc[2])
