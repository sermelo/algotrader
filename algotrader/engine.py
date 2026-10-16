"""Event-driven backtest engine for a long-only, single-instrument strategy."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .sizing import PositionSizer
from .strategy import SmaCrossStrategy


@dataclass
class Trade:
    """One completed round trip. Prices are fills (after slippage)."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    shares: int
    entry_price: float
    exit_price: float
    commission: float
    exit_reason: str          # 'signal', 'stop' or 'end'

    @property
    def pnl(self) -> float:
        """Net profit in dollars, after commissions (slippage is in the prices)."""
        return (self.exit_price - self.entry_price) * self.shares - self.commission


@dataclass
class BacktestResult:
    """Output of a backtest run."""
    initial_capital: float
    equity: pd.Series                       # mark-to-market equity at each bar close
    trades: list = field(default_factory=list)

    def trades_frame(self) -> pd.DataFrame:
        """Trades as a DataFrame (empty with the right columns if none)."""
        cols = ["entry_time", "exit_time", "shares", "entry_price",
                "exit_price", "commission", "exit_reason", "pnl"]
        rows = [{**{c: getattr(t, c) for c in cols[:-1]}, "pnl": t.pnl} for t in self.trades]
        return pd.DataFrame(rows, columns=cols)


class Backtester:
    """Runs a strategy over one instrument's bars.

    Execution rules (all designed to avoid look-ahead bias):
      * Signals are computed on a bar's close and filled at the NEXT bar's open.
      * Entries happen only on a fresh crossover (want_long goes False -> True),
        so a stopped-out position does not re-enter until the next crossover.
      * The stop is `atr_mult * ATR` below the entry fill, ATR taken from the
        signal bar. It is checked against each bar's LOW; if the bar opens below
        the stop (gap) the fill is the open, otherwise the stop price.
      * Any open position is closed at the last bar's close ('end').

    Args:
        strategy: provides indicators and the stop multiple.
        costs: commission/slippage model.
        sizer: position sizing rule.
        initial_capital: starting cash in dollars.
    """

    def __init__(self, strategy: SmaCrossStrategy = None, costs: CostModel = None,
                 sizer: PositionSizer = None, initial_capital: float = 10_000.0):
        self.strategy = strategy or SmaCrossStrategy()
        self.costs = costs or CostModel()
        self.sizer = sizer or PositionSizer()
        self.initial_capital = initial_capital

    def run(self, bars: pd.DataFrame) -> BacktestResult:
        """Backtest `bars` (indexed OHLCV, see BarLoader) and return the result."""
        df = self.strategy.indicators(bars)
        idx = df.index
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = df["atr"].to_numpy(float)
        want = df["want_long"].to_numpy(bool)
        n = len(df)

        cash = self.initial_capital
        shares = 0
        stop = entry_fill = entry_comm = 0.0
        entry_time = None
        pending = None                      # 'buy' / 'sell' decided at prior close
        pending_atr = np.nan
        equity = np.empty(n)
        trades = []

        def close_position(i, price, reason):
            nonlocal cash, shares
            fill = self.costs.sell_price(price)
            comm = self.costs.commission(shares)
            cash += shares * fill - comm
            trades.append(Trade(entry_time, idx[i], shares, entry_fill, fill,
                                entry_comm + comm, reason))
            shares = 0

        for i in range(n):
            # 1. Fill the order decided at the previous close, at this open.
            if pending == "buy" and shares == 0:
                fill = self.costs.buy_price(o[i])
                qty = self.sizer.shares(cash, fill, self.strategy.atr_mult * pending_atr)
                while qty > 0 and qty * fill + self.costs.commission(qty) > cash:
                    qty -= 1
                if qty > 0:
                    entry_comm = self.costs.commission(qty)
                    cash -= qty * fill + entry_comm
                    shares, entry_fill, entry_time = qty, fill, idx[i]
                    stop = fill - self.strategy.atr_mult * pending_atr
            elif pending == "sell" and shares > 0:
                close_position(i, o[i], "signal")
            pending = None

            # 2. Protective stop, checked against the bar's low.
            if shares > 0 and l[i] <= stop:
                close_position(i, min(o[i], stop), "stop")

            # 3. Mark to market, then 4. decide the next order from this close.
            equity[i] = cash + shares * c[i]
            if i + 1 < n:
                if shares > 0 and not want[i]:
                    pending = "sell"
                elif shares == 0 and want[i] and (i == 0 or not want[i - 1]) \
                        and not np.isnan(atr[i]):
                    pending, pending_atr = "buy", atr[i]

        if shares > 0:
            close_position(n - 1, c[-1], "end")
            equity[-1] = cash

        return BacktestResult(self.initial_capital, pd.Series(equity, index=idx), trades)
