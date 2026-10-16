"""algotrader: a small, tested backtester for long-only SMA-crossover strategies.

Main classes:
    BarLoader       -- reads and resamples OHLCV parquet files
    CostModel       -- commission and slippage
    PositionSizer   -- volatility-based position sizing
    SmaCrossStrategy-- computes indicators and entry/exit signals
    Backtester      -- runs a strategy over bars and returns a BacktestResult
    Metrics         -- performance statistics from a BacktestResult
"""
from .costs import CostModel
from .data import BarLoader
from .engine import Backtester, BacktestResult, Trade
from .metrics import Metrics
from .sizing import PositionSizer
from .strategy import SmaCrossStrategy

__all__ = ["BarLoader", "CostModel", "PositionSizer", "SmaCrossStrategy",
           "Backtester", "BacktestResult", "Trade", "Metrics"]
