import json

import numpy as np
import pandas as pd
import pytest

from algotrader.fundamentals import FundamentalSort
from algotrader.portfolio import CrossSectionalMomentum
from algotrader.volsubset import (COST_VARIANTS, FREQS, SIZES, UNIVERSES, VolatileSubset, make,
                                  n_configurations, report, strategies, verdicts, volatile)


def closes(days=300, n=9, seed=0):
    """S0-S2 calm (0.5% a day), S3-S5 medium (1.5%), S6-S8 wild (4%)."""
    rng = np.random.default_rng(seed)
    sd = np.repeat([0.005, 0.015, 0.04], n // 3)
    r = rng.normal(0.0005, 1, (days, n)) * sd
    idx = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"S{i}" for i in range(n)])


def test_volatile_keeps_the_most_volatile_third_from_past_returns_only():
    c = closes()
    keep = volatile(c, frac=1 / 3, lookback=20, min_periods=15)
    assert not keep.iloc[:15].any().any()                 # not enough returns yet
    assert (keep.iloc[60:].sum(axis=1) == 3).all()
    assert keep.iloc[60:][["S6", "S7", "S8"]].mean().min() > 0.9
    # later prices cannot change an earlier day's subset
    wrecked = c.copy()
    wrecked.iloc[200:] = wrecked.iloc[200:] * np.linspace(1, 50, 100)[:, None] * (np.arange(9) < 3)
    assert volatile(wrecked, 1 / 3, 20, 15).iloc[:200].equals(keep.iloc[:200])


def test_volatile_ignores_tickers_not_trading_that_day():
    c = closes()
    c.loc[c.index[100]:, "S8"] = np.nan
    keep = volatile(c, 1 / 3, 20, 15)
    assert not keep.iloc[100:]["S8"].any() and (keep.iloc[120:].sum(axis=1) == 2).all()  # a third of 8


def test_momentum_picks_only_inside_the_subset():
    c = closes()
    c["S0"] = 100 * 1.004 ** np.arange(len(c))            # calm and the best momentum
    inner = CrossSectionalMomentum(lookback=40, skip=0, top_n=2, every="M")
    plain, sub = inner.weights(c), VolatileSubset(inner, 1 / 3, 20, 15).weights(c)
    assert plain.iloc[-1]["S0"] == 0.5 and sub.iloc[-1]["S0"] == 0.0
    held = sub.iloc[60:]
    assert (held[[f"S{i}" for i in range(6)]] == 0).all().all()
    assert np.allclose(held.sum(axis=1), 1.0)
    # the same ranking rule within the subset: the best two of S6-S8 by momentum
    score = inner.score(c).iloc[-1][["S6", "S7", "S8"]]
    assert set(held.columns[held.iloc[-1] > 0]) == set(score.nlargest(2).index)


def test_fundamental_sort_uses_its_panel_and_direction():
    c = closes()
    panel = pd.DataFrame(np.tile(np.arange(9.0), (len(c), 1)), index=c.index, columns=c.columns)
    for largest, best in ((True, "S8"), (False, "S6")):
        inner = FundamentalSort(signal="x", top_n=1, largest=largest, every="M", panels={"x": panel})
        w = VolatileSubset(inner, 1 / 3, 20, 15).weights(c)
        assert w.iloc[-1][best] == 1.0


def test_validation():
    inner = CrossSectionalMomentum()
    for kw in ({"frac": 0}, {"frac": 1.5}, {"lookback": 1}, {"min_periods": 1}, {"lookback": 10, "min_periods": 20}):
        with pytest.raises(ValueError):
            VolatileSubset(inner, **kw)


def test_strategies_and_makers():
    s = strategies()
    assert sorted(s) == ["momentum", "op_profitability"]
    m = make(s["momentum"], None, "D", 5, "vol")
    assert isinstance(m, VolatileSubset) and m.inner.every == "D" and m.inner.top_n == 5
    assert make(s["momentum"], None, "W", 20, "all").every == "W"
    assert n_configurations() == 2 * 2 * 4


def fake(tmp_path):
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    rng = np.random.default_rng(3)
    cols, to = {}, {}
    for n in ("a", "b"):
        for u in UNIVERSES:
            for k in SIZES:
                for f in FREQS:
                    for v in COST_VARIANTS:
                        cols[f"{n}|{u}|top{k}|{f}|{v}"] = rng.normal(0.0004, 0.01, len(idx))
                    to[f"{n}|{u}|top{k}|{f}"] = 1.0
    df = pd.DataFrame(cols, index=idx)
    for k in SIZES:                                       # the filter helps `a` everywhere
        for f in FREQS:
            for v in COST_VARIANTS:
                df[f"a|vol|top{k}|{f}|{v}"] = df[f"a|all|top{k}|{f}|{v}"] + 0.002
    df["SPY"] = rng.normal(0.0004, 0.01, len(idx))
    df.to_csv(tmp_path / "returns.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({"volsubset": {"configurations": 16, "turnover": to}}))
    return df


def test_verdicts_find_a_planted_filter_gain(tmp_path):
    from algotrader.french import FACTORS
    df = fake(tmp_path)
    factors = pd.DataFrame(np.random.default_rng(9).normal(0, 0.01, (len(df), len(FACTORS))),
                           index=df.index, columns=list(FACTORS))
    v = verdicts(df, factors)
    assert v["filter_pass"] == ["a"]
    assert len(v["edge"]) == 12 and len(v["filter"]) == 12 and len(v["freq"]) == 8
    row = next(r for r in v["filter"] if r["strategy"] == "a" and r["label"] == "top5 M")
    assert abs(row["diff"] - 0.002 * 252) < 1e-9


def test_report(tmp_path):
    fake(tmp_path)
    text = report(tmp_path, data_dir=str(tmp_path / "no_french"))
    assert "Pre-registered rules" in text and "`a`" in text and "volatile third" in text
