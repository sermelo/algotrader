"""Portfolio and significance statistics on daily returns."""
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649


def portfolio_returns(curves: dict) -> pd.Series:
    """Daily returns of an equal-weight, daily-rebalanced portfolio of normalised curves.

    `curves` maps a name to a daily value Series; days a curve has no value are
    carried forward (no change).
    """
    df = pd.DataFrame(curves).sort_index().ffill()
    return df.pct_change().mean(axis=1).dropna()


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe of daily returns (risk-free 0); NaN if undefined."""
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS))


def returns_stats(r: pd.Series) -> dict:
    """Total return, CAGR, Sharpe and max drawdown of daily returns."""
    r = pd.Series(r).dropna()
    if r.empty:
        return {"return": float("nan"), "cagr": float("nan"), "sharpe": float("nan"),
                "max_drawdown": float("nan")}
    eq = (1 + r).cumprod()
    total = float(eq.iloc[-1] - 1)
    years = len(r) / TRADING_DAYS
    return {"return": total,
            "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else float("nan"),
            "sharpe": sharpe(r),
            "max_drawdown": float(min(0.0, (eq / eq.cummax() - 1).min()))}


def sharpe_interval(r: pd.Series, level: float = 0.9, block: int = 20, n: int = 1000,
                    seed: int = 0) -> tuple:
    """Circular block-bootstrap interval for the annualised Sharpe of daily returns.

    Blocks of `block` days keep short-range dependence. Returns (low, high); NaN if
    there are fewer than two blocks of data.
    """
    x = pd.Series(r).dropna().to_numpy()
    if len(x) < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nblocks = math.ceil(len(x) / block)
    stats = []
    for _ in range(n):
        starts = rng.integers(0, len(x), nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % len(x)
        s = x[idx[:len(x)]]
        sd = s.std(ddof=1)
        stats.append(s.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def alpha_beta(r: pd.Series, benchmark: pd.Series) -> dict:
    """OLS of daily strategy returns on benchmark returns (same days).

    Returns beta, annualised alpha (intercept * 252) and the alpha t-statistic.
    """
    df = pd.concat([pd.Series(r, name="r"), pd.Series(benchmark, name="b")], axis=1).dropna()
    if len(df) < 3 or df["b"].var() == 0:
        return {"beta": float("nan"), "alpha": float("nan"), "alpha_t": float("nan")}
    b = df["b"].to_numpy()
    y = df["r"].to_numpy()
    X = np.column_stack([np.ones_like(b), b])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - 2
    s2 = resid @ resid / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[0, 0]) if cov[0, 0] > 0 else float("nan")
    return {"beta": float(coef[1]), "alpha": float(coef[0] * TRADING_DAYS),
            "alpha_t": float(coef[0] / se) if se and se > 0 else float("nan")}


def expected_max_sharpe(n_trials: int, years: float) -> float:
    """Expected best annualised Sharpe among `n_trials` independent strategies with
    zero true edge, each measured over `years` (Bailey & Lopez de Prado, 2014).

    Uses the standard error of an annualised Sharpe near zero, 1 / sqrt(years). A
    result below this is what luck alone would produce from that many tries.
    """
    from statistics import NormalDist

    if n_trials < 2 or years <= 0:
        return 0.0
    z = NormalDist().inv_cdf
    se = 1 / math.sqrt(years)
    return se * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials)
                 + EULER_GAMMA * z(1 - 1 / (n_trials * math.e)))
