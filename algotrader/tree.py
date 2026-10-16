"""Decision-tree strategy after Naga et al. (2024), arXiv:2405.13959.

A per-stock decision tree classifies whether the next bar closes higher, from
nine technical features. See R&D/2405.13959-Decision-Trees-Intraday-Trading-Strategies.md.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import adx, rsi
from .strategy import average_true_range

FEATURES = ["ret_1", "ret_15", "rsi_14", "adx_14", "sma_ratio", "sma_corr",
            "vol_14", "vol_210", "vwap_ratio"]


def tree_features(bars: pd.DataFrame) -> pd.DataFrame:
    """The paper's nine inputs, each using only data up to the bar's close.

    The paper gives no window for the SMA/close correlation or the VWAP; 14 bars
    is used for both, and the VWAP uses the typical price (high + low + close) / 3.
    Undefined values (warm-up, zero volume, constant prices) are NaN.
    """
    close, volume = bars["close"], bars["volume"]
    ret_1 = close.pct_change(fill_method=None)
    ret_15 = close.pct_change(15, fill_method=None)
    sma = close.rolling(14).mean()
    typical = (bars["high"] + bars["low"] + close) / 3
    vwap = (typical * volume).rolling(14).sum() / volume.rolling(14).sum()
    feats = pd.DataFrame({
        "ret_1": ret_1,
        "ret_15": ret_15,
        "rsi_14": rsi(close, 14),
        "adx_14": adx(bars, 14),
        "sma_ratio": sma / close,
        "sma_corr": close.rolling(14).corr(sma),
        "vol_14": ret_1.rolling(14).std(),
        "vol_210": ret_15.rolling(14).std(),     # 14-bar volatility of 15-bar returns
        "vwap_ratio": vwap / close,
    }, index=bars.index)
    return feats.replace([np.inf, -np.inf], np.nan)


def next_bar_up(bars: pd.DataFrame) -> pd.Series:
    """Training label: 1.0 if the next close is above this close, else 0.0; NaN on the last bar."""
    nxt = bars["close"].shift(-1)
    return (nxt > bars["close"]).astype(float).where(nxt.notna())


@dataclass
class DecisionTreeStrategy:
    """Long while a per-stock decision tree predicts the next bar closes higher.

    Call `fit(train_bars)` on an earlier period, then backtest later bars. The
    tree is trained only on the bars passed to `fit`, so as long as those end
    before the backtest bars there is no look-ahead. Bars with any undefined
    feature are flat.

    Attributes:
        max_depth: tree depth (paper: 4).
        atr_period: ATR window for the engine's stop and sizing.
        atr_mult: stop distance in ATRs (the paper uses no stop; pass a large
            value to approximate that).
        random_state: seed for the tree's tie-breaking, for reproducibility.
    """
    max_depth: int = 4
    atr_period: int = 14
    atr_mult: float = 2.0
    random_state: int = 0
    model: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.atr_period < 1 or not self.atr_mult > 0:
            raise ValueError("require atr_period >= 1 and atr_mult > 0")

    def fit(self, bars: pd.DataFrame) -> "DecisionTreeStrategy":
        """Train the tree on `bars`; returns self."""
        from sklearn.tree import DecisionTreeClassifier

        X, y = tree_features(bars), next_bar_up(bars)
        ok = X.notna().all(axis=1) & y.notna()
        if y[ok].nunique() < 2:
            raise ValueError("training bars need both up and down moves after warm-up")
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, criterion="gini",
                                            random_state=self.random_state)
        self.model.fit(X[ok].to_numpy(), y[ok].to_numpy())
        return self

    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return `bars` plus the features, `atr` and the predicted `want_long`."""
        if self.model is None:
            raise RuntimeError("call fit() before indicators()")
        df = bars.copy()
        X = tree_features(bars)
        df[FEATURES] = X
        df["atr"] = average_true_range(df, self.atr_period)
        ok = X.notna().all(axis=1)
        want = pd.Series(False, index=df.index)
        if ok.any():
            want[ok] = self.model.predict(X[ok].to_numpy()) > 0.5
        df["want_long"] = want
        return df
