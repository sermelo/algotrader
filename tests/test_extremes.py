import numpy as np
import pandas as pd
import pytest

from algotrader import extremes


def days(n=300, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_month_ends_and_returns_with_a_delisting():
    idx = days(70)
    closes = pd.DataFrame({"A": np.linspace(100, 169, 70), "B": 100.0}, index=idx)
    closes.loc[idx[35]:, "B"] = np.nan                       # B's series ends in February
    closes.loc[idx[30]:idx[34], "B"] = 80.0
    ends = extremes.month_ends(idx)
    assert list(ends.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-07"]
    r = extremes.monthly_returns(closes, ends)
    a = closes["A"]
    assert r.loc["2020-01-31", "A"] == pytest.approx(a["2020-02-28"] / a["2020-01-31"] - 1)
    assert r.loc["2020-01-31", "B"] == pytest.approx(-0.2)  # to its last close
    assert np.isnan(r.loc["2020-02-28", "B"])               # no close at the start: not counted
    closes.loc[idx[40]:, "A"] = -5.0                         # broken adjustment: no return
    assert np.isnan(extremes.monthly_returns(closes, ends).loc["2020-02-28", "A"])


def test_price_features_are_causal():
    rng = np.random.default_rng(0)
    idx = days(400)
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (400, 3)), axis=0)), index=idx,
                          columns=list("ABC"))
    dv = closes * 1e6
    ends = extremes.month_ends(idx)
    f = extremes.price_features(closes, dv, ends)
    cut = ends[12]
    wrecked = closes.copy()
    wrecked.loc[wrecked.index > cut] *= 5
    g = extremes.price_features(wrecked, wrecked * 1e6, ends)
    for k in f:
        pd.testing.assert_frame_equal(f[k].loc[:cut], g[k].loc[:cut])
    e = ends[14]
    assert f["ret_1m"].loc[e, "A"] == pytest.approx(closes["A"][e] / closes["A"].shift(21)[e] - 1)
    assert f["max_ret_21"].loc[e, "B"] == pytest.approx(closes["B"].pct_change().loc[:e].tail(21).max())
    assert f["dist_52w_high"].loc[e].le(0).all()


def test_new_members_and_earnings_due():
    members = pd.DataFrame({"A": [True, True, True], "B": [False, False, True]},
                           index=pd.to_datetime(["2019-01-01", "2020-01-01", "2020-06-15"]))
    ends = pd.DatetimeIndex(["2020-03-31", "2020-07-31", "2021-07-30"])
    nm = extremes.new_members(members, ends)
    assert nm["B"].tolist() == [False, True, False] and not nm["A"].any()
    reaction = pd.DataFrame({"ticker": ["A", "B"], "day": pd.to_datetime(["2020-05-05", "2020-04-28"])})
    due = extremes.earnings_due(reaction, pd.DatetimeIndex(["2020-07-31", "2020-06-30"]), ["A", "B"])
    assert due.loc["2020-07-31"].tolist() == [True, False]   # May release -> due in August
    assert due.loc["2020-06-30"].tolist() == [False, True]   # April release -> due in July


def planted(months=60, n=100, seed=0):
    """Feature `up` makes a stock more likely to be a winner only; `vol` makes both tails likelier."""
    rng = np.random.default_rng(seed)
    ends = pd.date_range("2010-01-31", periods=months, freq="ME")
    tickers = [f"T{i}" for i in range(n)]
    up = pd.DataFrame(rng.normal(size=(months, n)), index=ends, columns=tickers)
    vol = pd.DataFrame(rng.uniform(0.02, 0.2, (months, n)), index=ends, columns=tickers)
    ret = rng.normal(size=(months, n)) * vol + 0.1 * (up > 1)
    members = pd.DataFrame(True, index=ends[:1], columns=tickers)
    members["T0"] = False                                    # never a member: excluded
    return pd.DataFrame(ret, index=ends, columns=tickers), members, {"up": up, "vol": vol}


def test_frame_groups_and_membership():
    ret, members, feats = planted()
    df = extremes.frame(ret, members, feats)
    assert "T0" not in set(df["ticker"])
    per = df.groupby("month")
    assert (per["top10"].sum() == 10).all() and (per["bottom10"].sum() == 10).all()
    assert per["excess"].mean().abs().max() < 1e-12
    row = df[(df["month"] == ret.index[3]) & (df["ticker"] == "T5")].iloc[0]
    assert row["ret"] == ret.iloc[3]["T5"] and row["up"] == feats["up"].iloc[3]["T5"]


def test_lift_finds_the_asymmetric_feature_and_not_the_symmetric_one():
    ret, members, feats = planted()
    df = extremes.frame(ret, members, feats)
    up = extremes.lift_table(df, "up").set_index("bucket")
    vol = extremes.lift_table(df, "vol").set_index("bucket")
    assert up.loc["Q5", "asymmetry"] > 0.3 and up.loc["Q5", "years_top_gt_bottom"] == up.loc["Q5", "years"]
    assert vol.loc["Q5", "lift_top_decile"] > 1.5 and vol.loc["Q5", "lift_bottom_decile"] > 1.5
    assert abs(vol.loc["Q5", "asymmetry"]) < 0.3
    assert vol.loc["Q1", "lift_top_decile"] < 0.5          # calm stocks rarely reach either tail
    assert extremes.spread(df, "up")["t"] > 3 and abs(extremes.spread(df, "vol")["t"]) < 3
    assert up["rows"].sum() == len(df)


def test_categories_are_their_own_buckets():
    ret, members, feats = planted(months=24)
    flag = feats["up"] > 1
    df = extremes.frame(ret, members, {"flag": flag})
    t = extremes.lift_table(df, "flag").set_index("bucket")
    assert set(t.index) == {"True", "False"} and t.loc["True", "lift_top_decile"] > 1.5


def test_sic_groups():
    assert extremes.sic_group(2834) == "chemicals_pharma" and extremes.sic_group("3674") == "tech_hardware"
    assert extremes.sic_group(3711) == "manufacturing" and extremes.sic_group(7372) == "software_services"
    assert extremes.sic_group(None) == "other" and extremes.sic_group(9999) == "other"
    assert extremes.sic_group(1311) == "energy_mining" and extremes.sic_group(6021) == "finance_realestate"
