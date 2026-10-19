import json

import numpy as np
import pandas as pd
import pytest

from algotrader.concentration import (BANDS, SIZES, BufferBand, RankBand, combinations, report, strategies, yearly_pick)
from algotrader.portfolio import CrossSectionalMomentum


def closes(n_stocks=8, days=300, seed=0):
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.002, -0.002, n_stocks)             # stock 0 rises fastest
    r = drift + rng.normal(0, 0.001, (days, n_stocks))
    idx = pd.date_range("2020-01-01", periods=days, freq="B")
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"S{i}" for i in range(n_stocks)])


def test_rank_band_holds_exactly_the_ranks_in_the_band():
    c = closes()
    make = lambda k: CrossSectionalMomentum(lookback=60, skip=0, top_n=k, every="M")
    last = lambda w: {t for t, v in w.iloc[-1].items() if v > 0}
    assert last(RankBand(make, 1, 3).weights(c)) == {"S0", "S1", "S2"}
    w = RankBand(make, 4, 5).weights(c)
    assert last(w) == {"S3", "S4"}
    assert w.iloc[-1][["S3", "S4"]].tolist() == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError):
        RankBand(make, 3, 2)


class Ranked:
    """Top-k weights from a fixed ranking per day (list of tickers, best first)."""

    def __init__(self, k, rankings, tickers):
        self.k, self.rankings, self.tickers = k, rankings, tickers

    def weights(self, closes):
        w = pd.DataFrame(0.0, index=closes.index, columns=self.tickers)
        for d, order in zip(closes.index, self.rankings):
            w.loc[d, order[:self.k]] = 1.0 / self.k
        return w


def test_buffer_keeps_a_holding_until_it_leaves_the_wider_band():
    t = [f"T{i:02d}" for i in range(15)]
    day0 = t
    day1 = t[:2] + t[3:7] + [t[2]] + t[7:]                     # T02 falls to rank 7 (inside the top 10)
    day2 = t[:2] + t[3:] + [t[2]]                              # T02 falls to rank 15 (out of the top 10)
    c = pd.DataFrame(1.0, index=pd.bdate_range("2024-01-01", periods=3), columns=t)
    make = lambda k: Ranked(k, [day0, day1, day2], t)
    w = BufferBand(make, 3, 10).weights(c)
    held = [set(w.columns[w.iloc[d] > 0]) for d in range(3)]
    assert held[0] == {"T00", "T01", "T02"}
    assert held[1] == {"T00", "T01", "T02"}                   # plain top 3 would have sold T02 for T03
    assert held[2] == {"T00", "T01", "T03"}                   # out of the top 10: replaced by the best new name
    assert w.iloc[2].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        BufferBand(make, 5, 3)


def test_yearly_pick_uses_only_earlier_years():
    idx = pd.bdate_range("2010-01-01", "2013-12-31")
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 3)), index=idx, columns=["a", "b", "c"])
    r.loc[r.index < "2012-01-01", "b"] += 0.01                # b was best before 2012
    combined, picks = yearly_pick(r, 1, pd.Timestamp("2012-01-01"))
    assert picks[2012] == ["b"]
    wrecked = r.copy()
    wrecked.loc[wrecked.index >= "2012-06-01", "c"] += 0.05   # later data cannot change the 2012 choice
    c2, p2 = yearly_pick(wrecked, 1, pd.Timestamp("2012-01-01"))
    assert p2[2012] == ["b"]
    pd.testing.assert_series_equal(combined[:"2012-12-31"], c2[:"2012-12-31"])


def fake_returns(tmp_path):
    idx = pd.bdate_range("2009-01-01", "2014-12-31")
    rng = np.random.default_rng(2)
    cols = {}
    for n in ("a", "b", "c", "d", "e", "f"):
        for k in SIZES:
            for var in ("base", "stress"):
                cols[f"{n}|top{k}|{var}"] = rng.normal(0.0004, 0.01, len(idx))
        for lo, hi in BANDS:
            cols[f"{n}|band{lo}-{hi}|base"] = rng.normal(0.0004, 0.01, len(idx))
    df = pd.DataFrame(cols, index=idx)
    df["SPY"] = rng.normal(0.0004, 0.01, len(idx))
    df.to_csv(tmp_path / "returns.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({"concentration": {
        "configurations": 59, "turnover": {f"{n}|top{k}": 1.0 for n in "abcdef" for k in SIZES}}}))
    return df


def test_combinations_and_report(tmp_path):
    df = fake_returns(tmp_path)
    comb, picks = combinations(df)
    assert set(comb) == {"all 6 strategies, top 5 each", "all 6 strategies, top 20 each",
                         "best 5 strategies (chosen yearly), top 5 each",
                         "best 5 strategies (chosen yearly), top 20 each", "best 1 strategy (chosen yearly), top 5"}
    expect = df[[f"{n}|top5|base" for n in "abcdef"]].mean(axis=1)
    pd.testing.assert_series_equal(comb["all 6 strategies, top 5 each"], expect, check_names=False)
    assert all(len(p) == 5 for p in picks[5].values())
    text = report(tmp_path, data_dir=str(tmp_path / "no_french"))
    assert "strategies pass" in text and "| 3 |" in text and "### `a`" in text


def test_eleven_strategies_accept_every_size():
    s = strategies()
    assert len(s) == 11
    for e in s.values():
        assert len(e.combos()) == len(SIZES)
