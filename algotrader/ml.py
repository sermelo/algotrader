"""Gradient-boosted classifier with a cost-aware label, trained across tickers.

The decision tree (tree.py) learned next-bar direction, an edge far smaller than
trading costs. Here the label asks a question costs cannot erase: does the close
`horizon` bars ahead exceed this close by more than `threshold`? One model is
trained on the pooled training bars of all tickers (features are scale-free), and
the strategy is long while the predicted probability exceeds `p_enter`.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import session_vwap
from .intraday import (bar_end_minutes, bars_per_session, previous_session_close,
                       session_open)
from .strategy import average_true_range
from .tree import FEATURES as TREE_FEATURES, tree_features

FEATURES = TREE_FEATURES + ["minute_of_day", "weekday", "ret_session", "gap",
                            "vwap_dist", "rel_volume", "ret_day1", "ret_day5", "ret_day20"]


def ml_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Tree features plus time of day, session context and multi-day returns.

    All causal. Multi-day returns count bars per full session at this bar size, so
    they mean the same on any bar size.
    """
    f = tree_features(bars)
    close = bars["close"]
    per_day = bars_per_session(bars.index)
    f["minute_of_day"] = bar_end_minutes(bars.index)
    f["weekday"] = bars.index.weekday
    f["ret_session"] = close / session_open(bars) - 1
    f["gap"] = session_open(bars) / previous_session_close(bars) - 1
    f["vwap_dist"] = close / session_vwap(bars) - 1
    f["rel_volume"] = bars["volume"] / bars["volume"].rolling(5 * per_day).mean()
    for days in (1, 5, 20):
        f[f"ret_day{days}"] = close.pct_change(days * per_day, fill_method=None)
    return f.replace([np.inf, -np.inf], np.nan)


def forward_label(bars: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """1.0 if close[t + horizon] / close[t] - 1 > threshold, else 0.0; NaN at the end."""
    fwd = bars["close"].shift(-horizon) / bars["close"] - 1
    return (fwd > threshold).astype(float).where(fwd.notna())


@dataclass
class GradientBoostingStrategy:
    """Long while P(forward return > threshold) > p_enter, from a pooled model.

    Call `fit_many([bars, ...])` (or `fit(bars)`) on training bars that end before
    the backtest period; labels near the end of each training frame are dropped
    because their future is not in the frame, so nothing leaks across the split.
    Missing feature values are handled natively by the model.
    """
    horizon: int = 4
    threshold: float = 0.003
    p_enter: float = 0.55
    max_depth: int = 3
    max_iter: int = 200
    learning_rate: float = 0.05
    atr_period: int = 14
    atr_mult: float = 50.0
    random_state: int = 0
    model: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.horizon < 1 or not 0 < self.p_enter < 1 or self.max_depth < 1:
            raise ValueError("require horizon >= 1, 0 < p_enter < 1, max_depth >= 1")

    def fit_many(self, frames) -> "GradientBoostingStrategy":
        from sklearn.ensemble import HistGradientBoostingClassifier

        X, y = [], []
        for bars in frames:
            label = forward_label(bars, self.horizon, self.threshold)
            ok = label.notna().to_numpy()
            X.append(ml_features(bars)[ok])
            y.append(label[ok])
        X, y = pd.concat(X), pd.concat(y)
        if y.nunique() < 2:
            raise ValueError("training labels need both classes")
        self.model = HistGradientBoostingClassifier(
            max_depth=self.max_depth, max_iter=self.max_iter, learning_rate=self.learning_rate,
            l2_regularization=1.0, random_state=self.random_state)
        self.model.fit(X[FEATURES].to_numpy(), y.to_numpy())
        return self

    def fit(self, bars: pd.DataFrame) -> "GradientBoostingStrategy":
        return self.fit_many([bars])

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit() or fit_many() before indicators()")
        df = bars.copy()
        X = ml_features(bars)
        df["p_up"] = self.model.predict_proba(X[FEATURES].to_numpy())[:, 1]
        df["want_long"] = df["p_up"] > self.p_enter
        df["atr"] = average_true_range(df, self.atr_period)
        return df
