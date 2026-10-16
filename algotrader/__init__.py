"""algotrader: a small, tested backtester for long-only strategies on minute bars.

Main classes:
    BarLoader       -- reads and resamples OHLCV parquet files
    CostModel       -- commission and slippage
    PositionSizer   -- volatility-based position sizing
    Strategy        -- protocol any strategy must satisfy (indicators + atr_mult)
    SmaCrossStrategy-- computes indicators and entry/exit signals
    DecisionTreeStrategy -- per-stock decision tree on technical features (fit, then backtest)
    Backtester      -- runs a strategy over bars and returns a BacktestResult
    Metrics         -- performance statistics from a BacktestResult
"""
from .costs import CostModel
from .data import BarLoader
from .engine import Backtester, BacktestResult, Trade
from .metrics import Metrics
from .sizing import PositionSizer
from .strategy import SmaCrossStrategy, Strategy, average_true_range, true_range
from .tree import DecisionTreeStrategy, tree_features

__all__ = ["BarLoader", "CostModel", "PositionSizer", "Strategy", "SmaCrossStrategy",
           "DecisionTreeStrategy", "tree_features", "average_true_range", "true_range",
           "Backtester", "BacktestResult", "Trade", "Metrics"]
