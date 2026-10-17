import numpy as np
import pandas as pd
import pytest

from algotrader.minutefeat import LABELS
from algotrader.minutemodel import WalkForward, best_of_day, information_coefficient, prepare, replay_fills, threshold_table

FEATS = ["f_good", "f_noise", "vol30"]


def planted(years=(2018, 2019, 2020, 2021), days=25, n=80, seed=0, signal=0.004):
    rng = np.random.default_rng(seed)
    parts = []
    for y in years:
        for d in pd.bdate_range(f"{y}-03-01", periods=days):
            for minute in (30, 60):
                good, noise, vol = rng.normal(size=n), rng.normal(size=n), rng.uniform(1, 20, n)
                ret = rng.normal(0, 0.001) + signal * good + 0.0005 * vol * rng.normal(size=n)
                df = pd.DataFrame({"date": d, "minute": float(minute), "ticker": [f"S{i}" for i in range(n)], "f_good": good, "f_noise": noise, "vol30": vol})
                for lab in LABELS:
                    df[lab] = ret
                parts.append(df)
    return pd.concat(parts, ignore_index=True)


def frames(df, **kw):
    p = prepare(df, derive=None, horizons=("fwd90",), **kw)
    return {y: g.reset_index(drop=True) for y, g in p.groupby(p["date"].dt.year)}


def test_prepare_keeps_the_volatile_tenth_of_each_decision_and_demeans_the_target():
    df = planted(years=(2018,), days=3)
    p = prepare(df, vol_rank=0.9, derive=None, horizons=("fwd90",))
    per = p.groupby(["date", "minute"]).size()
    assert per.between(8, 9).all()                                      # the top tenth of 80 stocks per decision (rank 0.9 and above)
    g = df.groupby(["date", "minute"])["fwd90"].transform("mean")
    assert np.allclose(p["x_fwd90"], (df["fwd90"] - g)[p.index])         # the mean is over all stocks, before the filter
    nine = df.groupby(["date", "minute"])["vol30"].transform(lambda s: s.nlargest(9).min())
    assert (df.loc[p.index, "vol30"] >= nine.loc[p.index]).all()


def test_walk_forward_learns_a_planted_signal_out_of_sample_and_only_from_earlier_years():
    df = planted()
    wf = WalkForward(first_test_year=2020, horizons=("fwd90",), features=FEATS, min_train_rows=100,
                     params=dict(learning_rate=0.1, max_iter=60, max_leaf_nodes=8, min_samples_leaf=20, l2_regularization=1.0, random_state=0))
    p = wf.run(frames(df, vol_rank=0.0))
    assert sorted(p["date"].dt.year.unique()) == [2020, 2021]
    assert information_coefficient(p, "fwd90").mean() > 0.15
    assert [row[0] for row in wf.log] == [2020, 2021]
    # wrecking the test years' targets cannot change a prediction made for an earlier test year
    wrecked = df.copy()
    wrecked.loc[wrecked["date"].dt.year == 2021, ["fwd90", "fwd30", "fwd240", "fwd10" if "fwd10" in wrecked else "fwd_close"]] = 5.0
    p2 = WalkForward(first_test_year=2020, horizons=("fwd90",), features=FEATS, min_train_rows=100,
                     params=wf.params).run(frames(wrecked, vol_rank=0.0))
    a = p[p["date"].dt.year == 2020].sort_values(["date", "minute", "ticker"]).reset_index(drop=True)
    b = p2[p2["date"].dt.year == 2020].sort_values(["date", "minute", "ticker"]).reset_index(drop=True)
    assert np.allclose(a["pred_fwd90"], b["pred_fwd90"])


def test_too_little_training_data_gives_no_prediction():
    df = planted(years=(2019, 2020), days=3, n=20)
    wf = WalkForward(first_test_year=2020, horizons=("fwd90",), features=FEATS, min_train_rows=10_000)
    p = wf.run(frames(df, vol_rank=0.0))
    assert p["pred_fwd90"].isna().all()


def test_threshold_table_counts_trades_and_nets_the_cost():
    df = planted(years=(2018,), days=10)
    rng = np.random.default_rng(1)
    df["pred_fwd90"] = df["f_good"] * 0.001 + rng.normal(0, 0.0001, len(df))
    df["fwd90"] = df["pred_fwd90"]                                      # a perfect forecast, so a higher threshold must pick bigger winners
    t = threshold_table(df, "fwd90", k=1, cost=0.0008, taus=(0.0, 0.002, 0.01))
    assert t["trades"].iloc[0] == 10 * 2                                # one pick per decision
    assert (np.diff(t["trades"]) <= 0).all() and t["trades"].iloc[2] == 0
    assert t["net_bp"].iloc[0] == pytest.approx(t["gross_bp"].iloc[0] - 8.0)
    assert t["gross_bp"].iloc[1] > t["gross_bp"].iloc[0]               # a higher threshold picks the better trades


def test_best_of_day_takes_the_single_most_confident_decision_each_day():
    df = planted(years=(2018,), days=12)
    df["pred_fwd90"] = df["f_good"] * 0.001
    df["fwd90"] = df["pred_fwd90"]
    t = best_of_day(df, "fwd90", cost=0.0008, taus=(0.0, 0.0015, 1.0))
    assert t["trades"].iloc[0] == 12 and t["trades"].iloc[2] == 0           # a trade per day, none above an unreachable threshold
    expected = df.groupby("date")["pred_fwd90"].max()
    assert t["gross_bp"].iloc[0] == pytest.approx(expected.mean() * 1e4)
    assert t["net_bp"].iloc[0] == pytest.approx(t["gross_bp"].iloc[0] - 8.0) and t["win"].iloc[0] > 0.9
    assert t["gross_bp"].iloc[1] >= t["gross_bp"].iloc[0]


def test_a_model_trained_on_shuffled_targets_finds_nothing():
    df = planted()
    kw = dict(first_test_year=2020, horizons=("fwd90",), features=FEATS, min_train_rows=100,
              params=dict(learning_rate=0.1, max_iter=60, max_leaf_nodes=8, min_samples_leaf=20, l2_regularization=1.0, random_state=0))
    real = information_coefficient(WalkForward(**kw).run(frames(df, vol_rank=0.0)), "fwd90").mean()
    placebo = information_coefficient(WalkForward(shuffle_seed=3, **kw).run(frames(df, vol_rank=0.0)), "fwd90").mean()
    assert real > 0.3 and abs(placebo) < 0.1 and abs(placebo) < real / 4


def test_replay_fills_hand_calculation():
    minutes = list(range(0, 200))
    bars = pd.DataFrame({"open": [100.0 + 0.1 * m for m in minutes], "high": [100.0 + 0.1 * m + 0.4 for m in minutes],
                         "low": [100.0 + 0.1 * m - 0.2 for m in minutes], "close": [100.0 + 0.1 * m + 0.05 for m in minutes]}, index=minutes)
    r = replay_fills(bars, d=30, hold=90)
    assert r["base"] == pytest.approx(112.0 / 103.0 - 1)                       # open at 30 -> open at 120
    assert r["worst_bar"] == pytest.approx((112.0 - 0.2) / (103.0 + 0.4) - 1)  # high of the entry bar, low of the exit bar
    assert r["delay2"] == pytest.approx(112.0 / 103.2 - 1) and r["delay5"] == pytest.approx(112.0 / 103.5 - 1)
    assert r["range_slip"] == pytest.approx((112.0 - 0.25 * 0.6) / (103.0 + 0.25 * 0.6) - 1)
    assert r["worst_bar"] < r["range_slip"] < r["base"]
    gap = bars.drop([30, 31])                                                     # no trade in the entry minute: no fill
    assert replay_fills(gap, 30, 90) is None
    skip = bars.drop([120, 121])
    assert replay_fills(skip, 30, 90)["base"] == pytest.approx(112.2 / 103.0 - 1)  # exits at the next bar with a trade
    assert replay_fills(bars, 150, 90)["base"] == pytest.approx((100.0 + 19.9 + 0.05) / 115.0 - 1)   # the session ends first: the last close
