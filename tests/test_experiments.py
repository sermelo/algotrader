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
    assert m["code_sha256"] == ex.experiment_fingerprint(TOY)
    assert m["package_sha256"] == ex.code_fingerprint()
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
    trades = pd.read_csv(out / "toy_trades.csv.gz")
    assert {"ticker", "fold", "entry_time", "exit_time", "pnl", "exit_reason"} <= set(trades.columns)
    assert set(trades["fold"]) <= {"2023-06-01", "2023-08-01", "2023-10-01"}
    assert pd.to_datetime(trades["entry_time"]).min() >= pd.Timestamp("2023-06-01")
    assert m["trade_log"]["written"] and m["trade_log"]["rows"] == len(trades)


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


def test_overlay_parameters_build_wrappers_and_load_market(setup, monkeypatch):
    data, out = setup
    from algotrader.overlays import MarketRegime, VolTargeted
    write_ticker(data, "SPY", list(np.linspace(100, 150, 300)))
    exp = ex.Experiment("ovl", "overlays", SmaCrossStrategy, "1D",
                        {"fast": 2, "slow": 8, "atr_period": 2, "vol_target": 0.1},
                        {"regime_days": [None, 20]}, costs=ex.CostModel(0, 0))
    assert isinstance(exp.make({"fast": 2, "slow": 8, "vol_target": 0.1}), VolTargeted)
    assert isinstance(exp.make({"fast": 2, "slow": 8, "regime_days": 20}), MarketRegime)
    assert exp.needs_market
    use(monkeypatch, exp)
    m = run(data, out, "ovl")
    assert m["params"]["regime_days"] in (None, 20)


def test_ticker_that_stops_trading_is_handled(setup):
    data, out = setup
    write_ticker(data, "CCC", wave(170, 30))                   # ends 2023-06-20
    ex.main(["toy", "--data-dir", str(data), "--tickers", "AAA", "BBB", "CCC",
             "--workers", "1", "--out", str(out)])
    folds = pd.read_csv(out / "toy_folds.csv")
    late = folds[folds["period"] == "fold:2023-10-01/base"]
    assert set(late["ticker"]) == {"AAA", "BBB"}               # CCC had no bars left
    early = folds[folds["period"] == "fold:2023-06-01/base"]
    assert "CCC" in set(early["ticker"])


def test_portfolio_experiment_walk_forward(setup, monkeypatch):
    data, out = setup
    write_ticker(data, "CCC", wave(300, 25, phase=0.6))
    from algotrader.portfolio import CrossSectionalMomentum
    exp = ex.PortfolioExperiment("xs", "toy momentum", CrossSectionalMomentum, "1D",
                                 {"skip": 0, "top_n": 1, "every": "W"},
                                 {"lookback": [5, 10]}, costs=ex.CostModel(0, 0))
    use(monkeypatch, exp)
    ex.main(["xs", "--data-dir", str(data), "--tickers", "AAA", "BBB", "CCC",
             "--workers", "1", "--out", str(out)])
    m = json.loads((out / "manifest.json").read_text())["xs"]
    assert m["kind"] == "portfolio" and m["params"]["lookback"] in (5, 10)
    assert set(m["walk_forward"]["fold_sharpe"]) == {"2023-06-01", "2023-08-01", "2023-10-01"}
    rows = pd.read_csv(out / "xs.csv")
    assert set(rows["period"]) == {"train", "test", "gross", "stress"}
    orders = pd.read_csv(out / "xs_trades.csv.gz")
    assert {"date", "ticker", "shares", "price", "fold"} <= set(orders.columns)
    assert (rows["ticker"] == "PORTFOLIO").all()
    assert "$100,000 total" in (out / "summary.md").read_text()


def test_worker_pool_uses_spawned_processes():
    # spawned (not forked) workers: safe after OpenMP threads exist in the parent
    assert ex._map(abs, [-1, -2, -3], workers=2) == [1, 2, 3]


def test_long_short_portfolio_experiment(setup, monkeypatch):
    data, out = setup
    from algotrader.portfolio import PairsStrategy
    exp = ex.PortfolioExperiment("pr", "toy pairs", PairsStrategy, "1D",
                                 {"pairs": (("AAA", "BBB"),)}, {"entry_z": [1.0, 1.5]},
                                 costs=ex.CostModel(0, 0), allow_short=True)
    use(monkeypatch, exp)
    m = run(data, out, "pr")
    assert m["kind"] == "portfolio" and m["params"]["entry_z"] in (1.0, 1.5)
    folds = pd.read_csv(out / "pr_folds.csv")
    assert len(folds) > 0


def test_fitted_portfolio_experiment_refits_per_fold(setup, monkeypatch):
    data, out = setup
    write_ticker(data, "CCC", wave(300, 25, phase=0.6))
    from algotrader.portfolio import CrossSectionalML
    exp = ex.PortfolioExperiment("xml", "toy ranking", CrossSectionalML, "1D",
                                 {"top_n": 1, "every": "W", "max_iter": 20},
                                 {"horizon": [3, 5]}, costs=ex.CostModel(0, 0), fit="pooled")
    use(monkeypatch, exp)
    ex.main(["xml", "--data-dir", str(data), "--tickers", "AAA", "BBB", "CCC",
             "--workers", "1", "--out", str(out)])
    m = json.loads((out / "manifest.json").read_text())["xml"]
    assert m["fit"] == "pooled" and set(m["fold_params"]) == {"2023-06-01", "2023-08-01",
                                                              "2023-10-01"}


def test_experiment_fingerprint_depends_only_on_used_code(monkeypatch, tmp_path):
    from algotrader.momentum import TurnOfMonthStrategy
    a = ex.Experiment("a", "", SmaCrossStrategy, "1D", {"fast": 2, "slow": 5})
    b = ex.Experiment("b", "", TurnOfMonthStrategy, "1D")
    fa, fb = ex.experiment_fingerprint(a), ex.experiment_fingerprint(b)
    assert fa == ex.experiment_fingerprint(a) and fa != fb
    assert ex.experiment_fingerprint(ex.Experiment("a", "", SmaCrossStrategy, "1D",
                                                   {"fast": 3, "slow": 5})) != fa
    # strategy modules: SMA does not depend on momentum.py, turn-of-month does
    deps = ex._module_deps("momentum", set())
    assert "momentum" in deps and "strategy" in deps and "tree" not in deps
    # the registry is not part of the runner source
    src = ex.runner_source()
    assert "Experiment(\"sma_daily\"" not in src and "def run_experiment" in src
