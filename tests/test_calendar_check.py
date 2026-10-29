import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("calendar_check", Path(__file__).parents[1] / "scripts/calendar_check.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def toy(years=6, seed=0):
    idx = pd.bdate_range("2012-01-02", f"{2011 + years}-12-31")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"A": rng.normal(0, 0.01, len(idx)), "SPY": rng.normal(0, 0.01, len(idx))}, index=idx)
    df.loc[df.index.month == 5, "A"] += 0.01             # A is better in May
    return df


def test_month_rule_learns_only_from_earlier_years():
    df = toy(years=7)
    rule, choices = cc.month_switch(df)
    assert rule.index.min().year == 2015 and choices[(2015, 5)] == "A"
    wrecked = df.copy()
    wrecked.loc[wrecked.index.year == 2017, "A"] = -0.05     # later data cannot change earlier choices
    _, again = cc.month_switch(wrecked)
    assert {k: v for k, v in again.items() if k[0] <= 2017} == {k: v for k, v in choices.items() if k[0] <= 2017}
    assert again[(2018, 1)] == "SPY"                         # ...but they change the next year's
    first_switch = next(d for d in rule.index if choices[(d.year, d.month)] != choices[(2015, 1)])
    pick = choices[(first_switch.year, first_switch.month)]
    assert rule[first_switch] == df.loc[first_switch, pick] - cc.SWITCH_COST


def test_month_table_and_shuffle_find_the_planted_month():
    df = toy(years=10)
    mo = cc.monthly(df)
    key = pd.Series(mo.index.month, index=mo.index)
    t = cc.group_table(mo, key, "month").set_index(["strategy", "month"])
    assert t.loc[("A", 5), "t"] > 3 and t.loc[("A", 5), "years_beat_spy"] >= 9
    _, p = cc.shuffle_test(mo, key, "A", np.random.default_rng(0))
    assert p < 0.05
