"""Multi-horizon trend following across asset classes (roadmap S2).

Hurst, Ooi & Pedersen, "A century of evidence on trend-following investing", JPM 2017:
an equal-weighted combination of 1, 3 and 12-month time-series momentum, each position
scaled by the asset's volatility. Moskowitz, Ooi & Pedersen (JFE 2012) for the
single-horizon version. Here on the ETFs in the universe (equity indices, sectors,
bonds, gold), as one portfolio.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import _rebalance_days

TREND_ASSETS = ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD",
                "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB")


@dataclass(frozen=True)
class MultiHorizonTrend:
    """For each asset in `assets`: signal = mean of sign(return over h days) for h in
    `horizons` (in [-1, 1]); weight = signal * `target_vol` / its annualised volatility
    over `vol_lookback` days, divided by the number of assets with a signal. With
    `long_only`, negative signals become 0 (cash). Rebalanced at period ends; the
    engine caps gross exposure (max_gross)."""
    horizons: tuple = (21, 63, 252)
    target_vol: float = 0.10
    vol_lookback: int = 63
    long_only: bool = False
    every: str = "M"
    assets: tuple = TREND_ASSETS

    def __post_init__(self):
        if not self.horizons or min(self.horizons) < 1 or self.target_vol <= 0:
            raise ValueError("require horizons >= 1 and target_vol > 0")
        if self.vol_lookback < 2:
            raise ValueError("require vol_lookback >= 2")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    def signal(self, closes: pd.DataFrame) -> pd.DataFrame:
        c = closes.ffill()
        s = sum(np.sign(c / c.shift(h) - 1) for h in self.horizons) / len(self.horizons)
        s = s.where(c.shift(max(self.horizons)).notna())
        return s.clip(lower=0) if self.long_only else s

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        sub = closes.copy()
        sub.loc[:, ~closes.columns.isin(self.assets)] = np.nan
        s = self.signal(sub)
        vol = sub.ffill().pct_change(fill_method=None).rolling(self.vol_lookback).std() * np.sqrt(252)
        raw = (s * self.target_vol / vol).where(sub.notna())
        n = raw.notna().sum(axis=1).clip(lower=1)
        w = raw.div(n, axis=0).fillna(0.0)
        rebal = _rebalance_days(closes.index, self.every)
        w[~rebal] = np.nan
        w.iloc[0] = w.iloc[0].fillna(0.0)
        return w.ffill().fillna(0.0)
