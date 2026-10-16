import numpy as np
import pandas as pd
import pytest

from algotrader.deep_arb import DeepArb, DeepArbModel, pca_residuals, stock_weights


def one_factor(T=400, N=12, seed=0, phi=0.0):
    """Returns = beta_i * market + idiosyncratic noise (AR(1) with coefficient phi)."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, T)
    b = rng.uniform(0.5, 1.5, N)
    e = np.zeros((T, N))
    z = rng.normal(0, 0.01, (T, N))
    for t in range(T):
        e[t] = z[t] + (phi * e[t - 1] if t else 0)
    idx = pd.bdate_range("2010-01-04", periods=T)
    return pd.DataFrame(m[:, None] * b + e, index=idx, columns=[f"S{i}" for i in range(N)]), e


def test_pca_residuals_remove_the_factor_and_are_causal():
    r, e = one_factor()
    eps, load = pca_residuals(r, k=1, window=100)
    assert eps.iloc[:100].isna().all().all()                    # needs a full window
    got = eps.iloc[100:].to_numpy()
    mkt = r.iloc[100:].mean(axis=1).to_numpy()
    corr_before = np.corrcoef(r.iloc[100:, 0], mkt)[0, 1]
    corr_after = np.corrcoef(got[:, 0], mkt)[0, 1]
    assert corr_before > 0.5 and abs(corr_after) < 0.3
    # prefix invariance: wrecking later returns does not change earlier residuals
    r2 = r.copy()
    r2.iloc[300:] *= 10
    eps2, _ = pca_residuals(r2, k=1, window=100)
    pd.testing.assert_frame_equal(eps.iloc[:300], eps2.iloc[:300])


def test_stock_weights_trade_the_residual_portfolio():
    r, _ = one_factor()
    eps, load = pca_residuals(r, k=1, window=100)
    d = r.index[200]
    cols, W, beta = load[d]
    a = np.linspace(-1, 1, len(cols))
    w = stock_weights(a, W, beta)
    assert np.abs(w).sum() == pytest.approx(1.0)
    raw = a - W @ (beta.T @ a)
    # w'R_t equals a'eps_t up to the normalisation
    assert w @ r.loc[d, cols].to_numpy() == pytest.approx(
        (a @ eps.loc[d, cols].to_numpy()) / np.abs(raw).sum())


def test_model_learns_planted_mean_reversion_out_of_sample():
    pytest.importorskip("torch")
    r, _ = one_factor(T=700, N=16, seed=1, phi=-0.6)
    eps, load = pca_residuals(r, k=1, window=60)
    m = DeepArbModel(k=1, lookback=5, train_days=300, refit_days=150, batch_days=50, epochs=15)
    w = m.weights_panel(r, eps, load)
    pnl = (w.shift(1) * r).sum(axis=1)                       # weights at d earn d+1
    live = pnl[w.abs().sum(axis=1).shift(1) > 0]
    assert len(live) > 200
    assert live.mean() / live.std() * np.sqrt(252) > 1.0      # planted edge is found
    assert (w.abs().sum(axis=1)[w.abs().sum(axis=1) > 0] - 1).abs().max() < 1e-6


def test_weights_do_not_see_the_future():
    pytest.importorskip("torch")
    r, _ = one_factor(T=520, N=10, seed=2, phi=-0.4)
    eps, load = pca_residuals(r, k=1, window=60)
    kw = dict(k=1, lookback=5, train_days=200, refit_days=100, batch_days=50, epochs=3)
    w1 = DeepArbModel(**kw).weights_panel(r, eps, load)
    r2 = r.copy()
    cut = r.index[420]
    r2.loc[r2.index > cut] *= -3
    eps2, load2 = pca_residuals(r2, k=1, window=60)
    w2 = DeepArbModel(**kw).weights_panel(r2, eps2, load2)
    pd.testing.assert_frame_equal(w1.loc[:cut], w2.loc[:cut])


def test_strategy_reads_panel():
    idx = pd.bdate_range("2013-01-01", periods=3)
    p = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [-0.5, -0.5, 0.0]}, index=idx)
    w = DeepArb(k=5, panels={"deep_arb_k5": p}).weights(pd.DataFrame(1.0, index=idx, columns=["A", "B", "C"]))
    assert w.loc[idx[0]].to_dict() == {"A": 0.5, "B": -0.5, "C": 0.0}
