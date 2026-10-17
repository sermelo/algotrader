import numpy as np
import pandas as pd
import pytest

from algotrader.chartcnn import (ChartCNN, ChartCNNModel, build_cnn, chart_batch, image_height,
                                 render_charts, windows)

D = 5


def window(closes, volumes=None, spread=0.0):
    c = np.array([closes], dtype=float)
    v = np.array([volumes if volumes is not None else [1.0] * len(closes)], dtype=float)
    return dict(o=c.copy(), h=c * (1 + spread), l=c * (1 - spread), c=c, v=v, ma=c.copy())


def test_shapes_and_values_are_binary():
    w = window([10, 11, 12, 11, 13])
    img = render_charts(**w)
    assert img.shape == (1, 1, image_height(D), 3 * D) and image_height(D) == 32
    assert set(np.unique(img)) <= {0, 255}


def test_close_pixel_moves_up_with_price_and_high_low_line_is_filled():
    w = window([10, 12, 14, 16, 18], spread=0.05)
    img = render_charts(**w)[0, 0] > 0
    price_rows = 32 - round(32 * 12 / 64) - 1
    closes = [int(np.flatnonzero(img[:price_rows, 3 * i + 2])[0]) for i in range(D)]
    assert closes == sorted(closes, reverse=True) and closes[0] > closes[-1]        # row 0 is the top
    for i in range(D):                                    # the middle column holds the whole high-low range
        col = np.flatnonzero(img[:price_rows, 3 * i + 1])
        assert col.max() - col.min() + 1 == len(col) and len(col) >= 2


def test_volume_bars_scale_to_the_window_maximum():
    w = window([10] * D, volumes=[1, 2, 4, 2, 0])
    img = render_charts(**w)[0, 0] > 0
    rows = round(32 * 12 / 64)
    heights = [int(img[32 - rows:, 3 * i + 1].sum()) for i in range(D)]
    assert heights == [round(rows * x / 4) for x in (1, 2, 4, 2, 0)]


def test_chart_has_no_scale_or_level():
    w = window([10, 12, 9, 15, 13], volumes=[3, 1, 4, 1, 5], spread=0.03)
    a = render_charts(**w)
    b = render_charts(**{k: (x * 7 if k != "v" else x * 1000) for k, x in w.items()})
    assert (a == b).all()


def test_chart_batch_uses_closes_before_the_window_for_the_moving_average():
    n = 40
    rng = np.random.default_rng(1)
    c = 50 + np.cumsum(rng.normal(0, 1, (n, 2)), axis=0).astype(np.float32)
    panels = {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.ones_like(c)}
    t = 20
    got = chart_batch(panels, np.array([t]), np.array([1]), D)
    ma = np.array([[c[t - D + 1 + i - (D - 1):t - D + 1 + i + 1, 1].mean() for i in range(D)]])
    g = lambda k: windows(panels[k], np.array([t]), np.array([1]), D)
    assert (got == render_charts(g("open"), g("high"), g("low"), g("close"), g("volume"), ma)).all()


def test_cnn_output_shape_for_each_chart_size():
    torch = pytest.importorskip("torch")
    for d in (5, 20):
        net = build_cnn(d, channels=(4, 8, 8))
        x = torch.zeros(3, 1, image_height(d), 3 * d)
        assert net(x).shape == (3, 1)


def planted(n_days=1800, n_tick=40, seed=0, h=5):
    """Prices where the next h days' direction follows the direction of the last 5 days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    r = rng.normal(0, 0.01, (n_days, n_tick))
    for t in range(5, n_days - h):
        past = r[t - 5:t].sum(axis=0)
        r[t:t + 1] += 0.006 * np.sign(past)               # momentum: follow the last week
    p = 50 * np.exp(np.cumsum(r, axis=0))
    frames = {f"S{i}": pd.DataFrame({"open": p[:, i], "high": p[:, i] * 1.005, "low": p[:, i] * 0.995,
                                     "close": p[:, i], "volume": 1e6 * (1 + 0.1 * rng.random(n_days))}, index=idx)
              for i in range(n_tick)}
    members = pd.DataFrame(True, index=[idx[0]], columns=list(frames))
    return frames, members


def test_cnn_finds_a_planted_pattern_out_of_sample():
    pytest.importorskip("torch")
    frames, members = planted()
    m = ChartCNNModel(d=5, h=5, channels=(4, 8, 8), train_years=3, stride=2, epochs=8, threads=1)
    s = m.scores(frames, members, first_year=2015)
    close = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    fwd = close.shift(-5) / close - 1
    both = pd.concat([s.stack(), fwd.stack()], axis=1).dropna()
    assert len(both) > 500
    assert both.corr(method="spearman").iloc[0, 1] > 0.15


def test_scores_unchanged_when_later_data_is_wrecked():
    pytest.importorskip("torch")
    frames, members = planted(n_days=1500, n_tick=20)
    m = ChartCNNModel(d=5, h=5, channels=(4, 8, 8), train_years=3, stride=3, epochs=2, threads=1)
    a = m.scores(frames, members, first_year=2014)
    first = a.dropna(how="all").index[0]                          # first scored month end
    rng = np.random.default_rng(9)
    wrecked = {}
    for t, f in frames.items():
        g = f.copy()
        later = g.index > first
        g.loc[later, ["open", "high", "low", "close"]] *= rng.uniform(0.2, 5, (later.sum(), 1))
        wrecked[t] = g
    b = ChartCNNModel(d=5, h=5, channels=(4, 8, 8), train_years=3, stride=3, epochs=2, threads=1) \
        .scores(wrecked, members, first_year=2014)
    assert np.allclose(a.loc[first].dropna(), b.loc[first].dropna(), atol=1e-5)


def test_membership_limits_the_scored_stocks():
    pytest.importorskip("torch")
    frames, members = planted(n_days=1500, n_tick=20)
    members = members.assign(**{c: False for c in list(frames)[10:]})
    m = ChartCNNModel(d=5, h=5, channels=(4, 8, 8), train_years=3, stride=3, epochs=1, threads=1)
    s = m.scores(frames, members, first_year=2014)
    assert s[list(frames)[10:]].isna().all().all() and s[list(frames)[:10]].notna().any().any()


def test_strategy_picks_top_and_bottom_and_validates():
    idx = pd.bdate_range("2020-01-01", periods=45)
    closes = pd.DataFrame(10.0, index=idx, columns=list("ABCD"))
    score = pd.DataFrame(np.nan, index=idx, columns=closes.columns)
    month_end = idx[idx.to_series().groupby(idx.to_period("M")).transform("max") == idx]
    score.loc[month_end[0]] = [0.9, 0.7, 0.3, 0.1]
    s = ChartCNN(top_n=1, short_n=1, panels={"chart_cnn": score})
    w = s.weights(closes)
    assert w.loc[month_end[0]].tolist() == [1.0, 0.0, 0.0, -1.0]
    assert w.loc[idx[0]].abs().sum() == 0                          # nothing before the first score
    with pytest.raises(ValueError):
        ChartCNN(top_n=0)
    with pytest.raises(ValueError):
        ChartCNN(every="W")


def test_long_experiments_are_registered_with_valid_grids_and_a_fingerprint():
    from algotrader import longrun, longrun_chart
    names = {e.name: e for e in longrun_chart.EXPERIMENTS}
    assert len(names["chart_cnn_long"].combos()) == 2 and len(names["chart_cnn_ls_long"].combos()) == 4
    assert all(e.first_fold == longrun_chart.FIRST_SCORE_YEAR + 1 for e in names.values())
    before = {n: longrun.fingerprint(e) for n, e in longrun.BY_NAME.items()}
    longrun_chart.main(["--list"])
    assert all(longrun.fingerprint(longrun.BY_NAME[n]) == f for n, f in before.items())   # others unchanged
    assert set(names) <= set(longrun.BY_NAME) and len(longrun.fingerprint(names["chart_cnn_long"])) == 64


def test_long_runner_end_to_end_on_a_toy_history(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from algotrader import longrun, longrun_chart
    frames, members = planted(n_days=1500, n_tick=24)
    spy = frames["S0"].copy()
    frames["SPY"] = spy
    monkeypatch.setattr(longrun_chart, "CHART", dict(d=5, h=5, channels=(4, 8, 8), train_years=3,
                                                     stride=5, epochs=2, patience=1))
    monkeypatch.setattr(longrun_chart, "FIRST_SCORE_YEAR", 2013)
    monkeypatch.chdir(tmp_path)
    longrun._PANELS.clear()
    panels = longrun_chart.chart_cnn_panels(frames, members)
    assert panels["panels"]["chart_cnn"].notna().any().any()
    assert (tmp_path / "data/long_panels/chart_cnn.parquet").exists()
    exp = longrun.LongExperiment("chart_cnn_long", "toy", ChartCNN, {"every": "M"}, {"top_n": [5, 10]},
                                 prepare=lambda f, m: {"panels": panels["panels"]}, first_fold=2014)
    mem = members.copy()
    res = longrun.run_long(exp, frames, mem)
    assert len(res["folds"]) == 2 and res["wf"]["base"].notna().any()
