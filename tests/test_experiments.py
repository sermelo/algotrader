import json

import numpy as np
import pandas as pd
import pytest

from algotrader import SmaCrossStrategy, experiments as ex
from helpers import make_bars


def write_ticker(path, ticker, closes):
    bars = make_bars(closes, start="2023-01-02", freq="1D")
    raw = bars.rename(columns=str.capitalize).rename_axis("datetime").reset_index()
    raw.to_parquet(path / f"{ticker}_clean.parquet")


def wave(n, period, amp=5.0, base=100.0, phase=0.0):
    return list(base + amp * np.sin(2 * np.pi * (np.arange(n) / period + phase)))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write_ticker(data, "AAA", wave(300, 40))
    write_ticker(data, "BBB", wave(300, 60, phase=0.3))
    exp = ex.Experiment("toy", "toy SMA", SmaCrossStrategy, "1D", {"atr_period": 2},
                        {"fast": [2, 5], "slow": [8]}, costs=ex.CostModel(0, 0))
    monkeypatch.setattr(ex, "TEST_START", "2023-08-01")
    monkeypatch.setattr(ex, "VALID_START", "2023-05-01")
    monkeypatch.setattr(ex, "EXPERIMENTS", [exp])
    monkeypatch.setattr(ex, "BY_NAME", {"toy": exp})
    return data, tmp_path / "results"


def run(data, out):
    ex.main(["toy", "--data-dir", str(data), "--tickers", "AAA", "BBB",
             "--workers", "1", "--out", str(out)])
    return json.loads((out / "manifest.json").read_text())["toy"]


def test_writes_results_manifest_and_summary(setup):
    data, out = setup
    m = run(data, out)
    rows = pd.read_csv(out / "toy.csv")
    assert sorted(rows["period"].unique()) == ["gross", "test", "train"]
    assert set(rows["ticker"]) == {"AAA", "BBB"}
    grid = pd.read_csv(out / "toy_grid.csv")
    assert len(grid) == 2
    best = json.loads(grid.loc[grid["portfolio_sharpe"].idxmax(), "params"])
    assert m["params"] == best
    assert m["code_sha256"] == ex.code_fingerprint()
    assert set(m["data"]) == {"AAA", "BBB"}
    equity = pd.read_csv(out / "toy_equity.csv", index_col=0, parse_dates=True)
    assert equity.index[1] >= pd.Timestamp("2023-08-01")     # [0] is the 1.0 start value
    assert "`toy`" in (out / "summary.md").read_text()


def test_parameter_choice_ignores_test_period(setup, tmp_path):
    data, out = setup
    chosen = run(data, out)["params"]
    # wreck the test period: the tuning result must not change
    bars = pd.read_parquet(data / "AAA_clean.parquet")
    late = bars["datetime"] >= "2023-08-01"
    bars.loc[late, ["Open", "High", "Low", "Close"]] *= 0.3
    bars.to_parquet(data / "AAA_clean.parquet")
    assert run(data, tmp_path / "again")["params"] == chosen


def test_portfolio_stats_equal_weight():
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    stats = ex.portfolio_stats({"a": pd.Series([1.0, 1.1, 1.21], index=idx),
                                "b": pd.Series([1.0, 1.0, 1.0], index=idx)})
    assert stats["return"] == pytest.approx(1.05 * 1.05 - 1)
    assert stats["max_drawdown"] == 0


def test_combos_skip_invalid_points():
    exp = ex.Experiment("x", "", SmaCrossStrategy, "1D", {}, {"fast": [5, 50], "slow": [20]})
    assert exp.combos() == [{"fast": 5, "slow": 20}]


def test_pooled_model_is_fitted_once_on_all_tickers(setup, monkeypatch):
    data, out = setup
    from algotrader.ml import GradientBoostingStrategy
    exp = ex.Experiment("pool", "pooled toy", GradientBoostingStrategy, "1D",
                        {"max_iter": 20, "atr_period": 2}, {"entry_quantile": [0.5, 0.6]},
                        costs=ex.CostModel(0, 0), fit="pooled")
    monkeypatch.setattr(ex, "EXPERIMENTS", [exp])
    monkeypatch.setattr(ex, "BY_NAME", {"pool": exp})
    ex.main(["pool", "--data-dir", str(data), "--tickers", "AAA", "BBB",
             "--workers", "1", "--out", str(out)])
    m = json.loads((out / "manifest.json").read_text())["pool"]
    assert m["fit"] == "pooled" and m["params"]["entry_quantile"] in (0.5, 0.6)
    assert len(pd.read_csv(out / "pool_grid.csv")) == 2


def test_low_activity_grid_points_cannot_be_chosen(setup, monkeypatch, tmp_path):
    data, out = setup
    exp = ex.Experiment("toy", "toy SMA", SmaCrossStrategy, "1D", {"atr_period": 2, "fast": 2},
                        {"slow": [3, 60]}, costs=ex.CostModel(0, 0))
    monkeypatch.setattr(ex, "BY_NAME", {"toy": exp})
    monkeypatch.setattr(ex, "EXPERIMENTS", [exp])
    run(data, out)
    grid = pd.read_csv(out / "toy_grid.csv")
    trades = sorted(grid["median_trades"])
    assert trades[0] < trades[1]                       # slow=60 trades far less than slow=3
    # require more trades than the less active point makes: only the busier one is eligible
    monkeypatch.setattr(ex, "MIN_TRADES", trades[1])
    m = run(data, tmp_path / "strict")
    busy = json.loads(grid.loc[grid["median_trades"].idxmax(), "params"])
    assert m["params"] == busy and m["selection_note"] == ""
    monkeypatch.setattr(ex, "MIN_TRADES", 10 ** 6)     # nobody qualifies: flagged
    assert "no grid point" in run(data, tmp_path / "none")["selection_note"]
