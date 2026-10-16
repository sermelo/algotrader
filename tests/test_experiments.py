import json

import numpy as np
import pandas as pd
import pytest

from algotrader import SmaCrossStrategy, experiments as ex
from algotrader.engine import Backtester
from helpers import make_bars


def write_ticker(path, ticker, closes):
    bars = make_bars(closes, start="2023-01-02", freq="1D")
    raw = bars.rename(columns=str.capitalize).rename_axis("datetime").reset_index()
    raw.to_parquet(path / f"{ticker}_clean.parquet")


def wave(n, period, amp=5.0, base=100.0, phase=0.0):
    return list(base + amp * np.sin(2 * np.pi * (np.arange(n) / period + phase)))


def use(monkeypatch, *exps):
    monkeypatch.setattr(ex, "EXPERIMENTS", list(exps))
    monkeypatch.setattr(ex, "BY_NAME", {e.name: e for e in exps})


TOY = ex.Experiment("toy", "toy SMA", SmaCrossStrategy, "1D", {"atr_period": 2},
                    {"fast": [2, 5], "slow": [8]}, costs=ex.CostModel(0, 0))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_ticker(data, "AAA", wave(300, 40))           # 2023-01-02 .. 2023-10-28
    write_ticker(data, "BBB", wave(300, 60, phase=0.3))
    monkeypatch.setattr(ex, "TEST_START", "2023-08-01")
    monkeypatch.setattr(ex, "FOLD_STARTS", ["2023-06-01", "2023-08-01", "2023-10-01"])
    monkeypatch.setattr(ex, "VALID_MONTHS", 2)
    use(monkeypatch, TOY)
    return data, tmp_path / "results"


def run(data, out, name="toy"):
    ex.main([name, "--data-dir", str(data), "--tickers", "AAA", "BBB",
             "--workers", "1", "--out", str(out)])
    return json.loads((out / "manifest.json").read_text())[name]


def test_writes_results_manifest_and_summary(setup):
    data, out = setup
    m = run(data, out)
    rows = pd.read_csv(out / "toy.csv")
    assert sorted(rows["period"].unique()) == ["gross", "stress", "test", "train"]
    assert set(rows["ticker"]) == {"AAA", "BBB"}
    grid = pd.read_csv(out / "toy_grid.csv")
    assert sorted(grid["fold"].unique()) == ["2023-06-01", "2023-08-01", "2023-10-01"]
    at_test = grid[grid["fold"] == "2023-08-01"]
    assert m["params"] == json.loads(at_test.loc[at_test["portfolio_sharpe"].idxmax(), "params"])
    assert set(m["fold_params"]) == {"2023-06-01", "2023-08-01", "2023-10-01"}
    assert m["code_sha256"] == ex.code_fingerprint()
    assert set(m["data"]) == {"AAA", "BBB"}
    w = m["walk_forward"]
    assert set(w["fold_sharpe"]) == {"2023-06-01", "2023-08-01", "2023-10-01"}
    wf = pd.read_csv(out / "toy_wf_returns.csv", index_col=0, parse_dates=True)
    assert wf.index[0] >= pd.Timestamp("2023-06-01")             # out of sample only
    assert {"base", "gross", "stress", "hold"} <= set(wf.columns)
    folds = pd.read_csv(out / "toy_folds.csv")
    assert "fold:2023-06-01/base" in set(folds["period"])
    equity = pd.read_csv(out / "toy_equity.csv", index_col=0, parse_dates=True)
    assert equity.index[1] >= pd.Timestamp("2023-08-01")          # [0] is the 1.0 start value
    text = (out / "summary.md").read_text()
    assert "`toy`" in text and "Walk-forward" in text


def test_parameter_choice_ignores_data_after_each_fold_start(setup, tmp_path):
    data, out = setup
    first = run(data, out)["fold_params"]
    # wreck everything from 2023-08-01 on: choices made at or before that date must not change
    bars = pd.read_parquet(data / "AAA_clean.parquet")
    late = bars["datetime"] >= "2023-08-01"
    bars.loc[late, ["Open", "High", "Low", "Close"]] *= 0.3
    bars.to_parquet(data / "AAA_clean.parquet")
    again = run(data, tmp_path / "again")["fold_params"]
    assert again["2023-06-01"] == first["2023-06-01"]
    assert again["2023-08-01"] == first["2023-08-01"]


def test_slice_result_rebases_capital_and_keeps_window_trades():
    bars = make_bars([10, 10, 11, 12, 13, 12, 11, 10, 10, 10], freq="1D")
    res = Backtester(SmaCrossStrategy(2, 3, 2, 50.0), ex.CostModel(0, 0)).run(bars)
    part = ex.slice_result(res, bars.index[4], bars.index[8])
    assert part.initial_capital == res.equity.iloc[3]
    assert len(part.equity) == 4
    assert all(bars.index[4] <= t.entry_time < bars.index[8] for t in part.trades)
    assert ex.slice_result(res, pd.Timestamp("2030-01-01")) is None


def test_stress_costs_widen_the_spread():
    assert TOY.costs_for("gross").spread_bps == 0
    assert ex.BASE_COSTS.spread_bps == ex.SPREAD_BPS
    exp = ex.Experiment("x", "", SmaCrossStrategy, "1D")
    assert exp.costs_for("stress").spread_bps == ex.STRESS_SPREAD_BPS
    assert exp.costs_for("stress").min_commission == exp.costs.min_commission


def test_pooled_model_is_fitted_once_on_all_tickers(setup, monkeypatch):
    data, out = setup
    from algotrader.ml import GradientBoostingStrategy
    exp = ex.Experiment("pool", "pooled toy", GradientBoostingStrategy, "1D",
                        {"max_iter": 20, "atr_period": 2}, {"entry_quantile": [0.5, 0.6]},
                        costs=ex.CostModel(0, 0), fit="pooled")
    use(monkeypatch, exp)
    m = run(data, out, "pool")
    assert m["fit"] == "pooled" and m["params"]["entry_quantile"] in (0.5, 0.6)
    assert len(pd.read_csv(out / "pool_grid.csv")) == 2 * 3          # 2 points x 3 folds


def test_combos_skip_invalid_points():
    exp = ex.Experiment("x", "", SmaCrossStrategy, "1D", {}, {"fast": [5, 50], "slow": [20]})
    assert exp.combos() == [{"fast": 5, "slow": 20}]


def test_low_activity_grid_points_cannot_be_chosen(setup, monkeypatch, tmp_path):
    data, out = setup
    exp = ex.Experiment("toy", "toy SMA", SmaCrossStrategy, "1D", {"atr_period": 2, "fast": 2},
                        {"slow": [3, 60]}, costs=ex.CostModel(0, 0))
    use(monkeypatch, exp)
    run(data, out)
    grid = pd.read_csv(out / "toy_grid.csv")
    g = grid[grid["fold"] == "2023-08-01"]
    trades = sorted(g["median_trades"])
    assert trades[0] < trades[1]                       # slow=60 trades far less than slow=3
    monkeypatch.setattr(ex, "MIN_TRADES", trades[1])
    m = run(data, tmp_path / "strict")
    busy = json.loads(g.loc[g["median_trades"].idxmax(), "params"])
    assert m["params"] == busy and m["selection_note"] == ""
    monkeypatch.setattr(ex, "MIN_TRADES", 10 ** 6)     # nobody qualifies: flagged
    assert "no grid point" in run(data, tmp_path / "none")["selection_note"]
