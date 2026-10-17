import json

import numpy as np
import pandas as pd
import pytest

from algotrader import longrun
from algotrader.history import parse_members
from algotrader.portfolio import CrossSectionalMomentum


def _closes(n=10):
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame({"A": np.linspace(100, 200, n), "B": np.linspace(100, 110, n),
                         "C": np.linspace(100, 300, n)}, index=idx)


def test_point_in_time_hides_non_members():
    closes = _closes()
    # C is the best performer but only joins the index on day 6; membership lists are sparse
    members = longrun.membership_matrix(parse_members(
        'date,tickers\n2010-01-04,"A,B"\n2010-01-11,"A,B,C"\n'))
    strat = longrun.PointInTime(CrossSectionalMomentum(lookback=2, skip=0, top_n=1, every="D"),
                                members)
    w = strat.weights(closes)
    assert w.loc["2010-01-08"].to_dict() == {"A": 1.0, "B": 0.0, "C": 0.0}   # C not yet a member
    assert w.loc["2010-01-13"]["C"] == 1.0                                 # member from 01-11


def test_fold_starts_and_turnover():
    idx = pd.bdate_range("2010-01-04", "2014-06-30")
    assert longrun.fold_starts(idx, first=2012) == [pd.Timestamp("2012-01-01"),
                                                     pd.Timestamp("2013-01-01"),
                                                     pd.Timestamp("2014-01-01")]

    class R:
        equity = pd.Series(100_000.0, index=pd.bdate_range("2010-01-04", periods=252))
        orders = [(pd.Timestamp("2010-02-01"), "A", 500, 100.0, 1.0),
                  (pd.Timestamp("2010-03-01"), "A", -500, 100.0, 1.0)]

    assert longrun.turnover(R()) == pytest.approx(0.5)   # 100k traded / 2 / 100k / 1 year


def _history(tmp_path):
    """Three member tickers over 2008-2013 in WIKI format, plus SPY; one ticker leaves."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2008-01-02", "2013-12-31")
    (tmp_path / "wiki").mkdir(parents=True)
    for t, drift in (("AAA", 0.0008), ("BBB", 0.0), ("CCC", -0.0004), ("SPY", 0.0003)):
        p = 50 * np.cumprod(1 + drift + rng.normal(0, 0.01, len(idx)))
        df = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1e6}, index=idx)
        if t == "CCC":
            df = df[df.index < "2012-06-01"]                      # delisted mid-2012
        df.to_parquet(tmp_path / "wiki" / f"{t}.parquet")
    (tmp_path / "sp500_members.csv").write_text(
        'date,tickers\n2008-01-02,"AAA,BBB,CCC"\n2012-06-01,"AAA,BBB"\n')
    return tmp_path


def test_run_long_end_to_end(tmp_path):
    hist = _history(tmp_path / "h")
    frames = longrun.load_frames(hist)
    assert set(frames) == {"AAA", "BBB", "CCC", "SPY"}
    members = longrun.membership_matrix(parse_members((hist / "sp500_members.csv").read_text()))
    exp = longrun.LongExperiment("toy", "toy", CrossSectionalMomentum, {"every": "M"},
                                 {"lookback": [63, 126], "skip": [0], "top_n": [1]})
    res = longrun.run_long(exp, frames, members)
    assert list(res["folds"]["fold"]) == ["2012-01-01", "2013-01-01"]
    wf = res["wf"]
    assert wf.index.min() >= pd.Timestamp("2012-01-01")
    assert {"base", "gross", "stress", "hold"} <= set(wf.columns)
    assert (wf["gross"].sum() >= wf["base"].sum() >= wf["stress"].sum())
    # choice at a fold uses only data before it: the grid rows of 2012 score data < 2012
    held = longrun.PointInTime(exp.make(json.loads(res["folds"]["params"][0])), members)
    assert "SPY" not in held.weights(pd.DataFrame({t: f["close"] for t, f in frames.items()})).columns[
        held.weights(pd.DataFrame({t: f["close"] for t, f in frames.items()})).abs().sum() > 0]
    longrun.write(res, tmp_path / "out", hist)
    longrun.BY_NAME["toy"] = exp
    try:
        longrun.write_summary(tmp_path / "out")
    finally:
        del longrun.BY_NAME["toy"]
    text = (tmp_path / "out" / "summary.md").read_text()
    assert "`toy`" in text and "(stale)" not in text


def test_choice_ignores_data_after_fold(tmp_path):
    hist = _history(tmp_path / "h")
    frames = longrun.load_frames(hist)
    members = longrun.membership_matrix(parse_members((hist / "sp500_members.csv").read_text()))
    exp = longrun.LongExperiment("toy", "toy", CrossSectionalMomentum, {"every": "M"},
                                 {"lookback": [63, 126], "skip": [0], "top_n": [1]})
    base = longrun.run_long(exp, frames, members)
    wrecked = {t: f.copy() for t, f in frames.items()}
    for f in wrecked.values():
        f.loc[f.index >= "2012-01-01", ["open", "high", "low", "close"]] *= np.linspace(
            1, 5, (f.index >= "2012-01-01").sum())[:, None]
    other = longrun.run_long(exp, wrecked, members)
    assert other["folds"]["params"][0] == base["folds"]["params"][0]


def test_load_frames_skips_reused_tickers(tmp_path):
    hist = _history(tmp_path / "h")
    (hist / "tiingo").mkdir()
    idx = pd.bdate_range("2025-01-02", periods=50)       # a later company reusing the ticker
    pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                 index=idx).to_parquet(hist / "tiingo" / "AAA.parquet")
    pd.DataFrame({"status": ["partial"], "cover": [0.0]}, index=["AAA"]).to_csv(hist / "coverage.csv")
    frames = longrun.load_frames(hist)
    assert frames["AAA"].index.min() < pd.Timestamp("2009-01-01")   # WIKI used instead


def test_load_frames_prefers_tiingo_then_hf_then_wiki(tmp_path):
    hist = _history(tmp_path / "h")
    (hist / "hf").mkdir()
    (hist / "tiingo").mkdir()
    idx = pd.bdate_range("2008-01-02", periods=100)
    for folder, t, price in (("hf", "AAA", 7.0), ("hf", "BBB", 8.0), ("tiingo", "BBB", 9.0)):
        pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0},
                     index=idx).to_parquet(hist / folder / f"{t}.parquet")
    frames = longrun.load_frames(hist)
    assert frames["AAA"]["close"].iloc[0] == 7.0          # HF beats WIKI
    assert frames["BBB"]["close"].iloc[0] == 9.0          # Tiingo beats HF
    assert frames["CCC"]["close"].iloc[0] != 7.0          # no HF: WIKI
    assert longrun.load_frames(hist, use_hf=False)["AAA"]["close"].iloc[0] != 7.0
