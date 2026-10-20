import json

import numpy as np
import pandas as pd

from algotrader.portfolio import CrossSectionalMomentum
from algotrader.rebalance import COST_VARIANTS, FREQS, SIZES, compare, jobs, maker, n_configurations, report, strategies


def closes(days=400, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.02, (days, 12))
    idx = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"S{i}" for i in range(12)])


class Exp:
    strategy = CrossSectionalMomentum
    fixed = {"every": "M", "lookback": 20, "skip": 0}


def test_weekly_changes_holdings_only_at_week_ends_and_more_often_than_monthly():
    c = closes()
    wm = maker(Exp, None, "M")(3).weights(c)
    ww = maker(Exp, None, "W")(3).weights(c)
    changes = lambda w: w.index[1:][(w.diff().abs().sum(axis=1) > 0)[1:]]
    cm, cw = changes(wm), changes(ww)
    assert len(cw) > len(cm)
    week_end = pd.Series(c.index.to_period("W"), index=c.index)
    last = set(c.index[(week_end != week_end.shift(-1)).to_numpy()])
    assert set(cw) <= last  # decided at a week's last close (traded at the next open)
    assert (wm.sum(axis=1).iloc[-1], ww.sum(axis=1).iloc[-1]) == (1.0, 1.0)


def test_jobs_and_strategies():
    js = jobs(maker(Exp, None, "W"))
    assert [j[0] for j in js] == ["top5", "top20", "buf5-20"]
    s = strategies()
    assert len(s) == 10 and "seasonality" not in s
    for e in s.values():
        assert maker(e, {}, "W")(5).every == "W"  # no duplicate `every`
    assert n_configurations() == 10 * 2 * 3


def fake(tmp_path):
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    rng = np.random.default_rng(3)
    cols, to = {}, {}
    for n in ("a", "b", "c"):
        for lab in [f"top{k}" for k in SIZES] + ["buf5-20"]:
            for f in FREQS:
                for v in (COST_VARIANTS if lab.startswith("top") else ("base", "stress")):
                    cols[f"{n}|{lab}|{f}|{v}"] = rng.normal(0.0004, 0.01, len(idx))
                to[f"{n}|{lab}|{f}"] = 1.0
    df = pd.DataFrame(cols, index=idx)
    df["a|top5|W|base"] = df["a|top5|M|base"] + 0.002          # weekly much better for `a`
    df["SPY"] = rng.normal(0.0004, 0.01, len(idx))
    df.to_csv(tmp_path / "returns.csv")
    (tmp_path / "manifest.json").write_text(json.dumps({"rebalance": {"configurations": 18, "turnover": to}}))
    return df


def test_compare_measures_weekly_minus_monthly(tmp_path):
    df = fake(tmp_path)
    cmp = compare(df)
    row = cmp[(cmp.strategy == "a") & (cmp.label == "top5")].iloc[0]
    assert abs(row["diff"] - 0.002 * 252) < 1e-9 and row["diff_t"] > 10
    assert len(cmp) == 3 * 3


def test_report(tmp_path):
    fake(tmp_path)
    text = report(tmp_path, data_dir=str(tmp_path / "no_french"))
    assert "Average over the 3 strategies" in text and "| top5 |" in text and "`a`" in text
