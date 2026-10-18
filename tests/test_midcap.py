import numpy as np
import pytest
import pandas as pd

from algotrader import longrun, midcap
from algotrader.midcap import equal_weight, load, segments


def frame(dates, closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000.0},
                        index=pd.DatetimeIndex(pd.to_datetime(dates)))


def test_segments_cut_long_gaps_and_gapped_jumps_only():
    d = list(pd.bdate_range("2020-01-01", periods=10)) + list(pd.bdate_range("2020-06-01", periods=10))
    parts = segments(frame(d, [10] * 20))
    assert [len(p) for p in parts] == [10, 10]                     # 5-month gap: a new company
    d2 = list(pd.bdate_range("2020-01-01", periods=10)) + list(pd.bdate_range("2020-01-27", periods=10))
    assert [len(p) for p in segments(frame(d2, [10] * 10 + [40] * 10))] == [10, 10]   # 2-week halt, price x4
    assert len(segments(frame(d2, [10] * 10 + [11] * 10))) == 1    # same halt, small move: same company
    d3 = pd.bdate_range("2020-01-01", periods=20)
    assert len(segments(frame(d3, [10] * 10 + [25] * 10))) == 1    # x2.5 without a halt: kept (real news)
    assert [len(p) for p in segments(frame(d3, [10] * 10 + [60] * 10))] == [10, 10]   # x6 in a day: never a return


def test_load_names_pieces_and_copies_membership(tmp_path):
    prices = tmp_path / "hf"
    prices.mkdir()
    d = list(pd.bdate_range("2012-01-02", periods=80)) + list(pd.bdate_range("2013-01-02", periods=80))
    frame(d, [10] * 160).to_parquet(prices / "AB_C.parquet")
    frame(pd.bdate_range("2012-01-02", periods=30), [5] * 30).to_parquet(prices / "SHORT.parquet")
    frame(pd.bdate_range("2012-01-02", periods=80), [5] * 79 + [-1]).to_parquet(prices / "NEG.parquet")
    frame(pd.bdate_range("2012-01-02", periods=80), [5] * 80).to_parquet(prices / "ZVZZT.parquet")
    frame(pd.bdate_range("2012-01-02", periods=80), [0.5] * 80).to_parquet(prices / "PENNY.parquet")
    (tmp_path / "m.csv").write_text('date,tickers\n2012-01-31,"AB.C,SHORT,NOFILE,NEG,ZVZZT,PENNY"\n2013-01-31,"AB.C"\n')
    frames, members = load(tmp_path / "m.csv", prices, tmp_path / "none.parquet", start="2008-01-01")
    assert sorted(frames) == ["AB.C", "AB.C~2"]                    # too few days, no file, negative price, test symbol, adjusted pennies
    assert sorted(members["ticker"].unique()) == ["AB.C", "AB.C~2"] and len(members) == 4


def test_equal_weight_uses_previous_day_members():
    idx = pd.bdate_range("2020-01-01", periods=4)
    frames = {"A": frame(idx, [10, 11, 11, 11]), "B": frame(idx, [10, 10, 12, 12])}
    m = pd.DataFrame({"A": [True, True], "B": [False, True]}, index=pd.to_datetime(["2019-12-31", "2020-01-02"]))
    ew = equal_weight(frames, m)
    assert ew.iloc[1] == pytest.approx(0.1)                                       # day 2: only A was a member on day 1
    assert ew.iloc[2] == pytest.approx((0.0 + 0.2) / 2)                           # day 3: both


def test_experiments_costs_and_fingerprints():
    exps = midcap.experiments()
    assert [e.name for e in exps] == [n.replace("_long", "_mid") for n in midcap.BASE]
    assert all(e.name not in longrun.BY_NAME for e in exps)
    e = exps[0]
    assert midcap.fingerprint(e) != longrun.fingerprint(e)          # the mid-cap code and costs are covered
    assert midcap.COSTS["base"].spread_bps == 5.0 and midcap.COSTS["stress"].spread_bps == 10.0
    assert longrun.COSTS["base"].spread_bps == 2.0                  # importing does not change the long runs
