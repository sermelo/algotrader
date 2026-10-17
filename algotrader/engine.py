"""Event-driven backtest engine for a long-only, single-instrument strategy."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .sizing import PositionSizer
from .strategy import SmaCrossStrategy, Strategy


@dataclass
class Trade:
    """One completed round trip. Prices are fills (after slippage)."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    shares: int
    entry_price: float
    exit_price: float
    commission: float
    exit_reason: str          # 'signal', 'stop', 'target', 'close' (exit_at_close) or 'end'

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
    position: pd.Series | None = None       # shares held at each bar close

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

    Optional strategy columns:
      * `stop_price`: absolute stop level decided on the signal bar. When present
        and not NaN it replaces the ATR stop, and sizing uses (entry fill - stop).
        If the fill is already at or below it, no position is opened.
      * `exit_at_close`: on bars where it is True, an open position is closed at
        that bar's close ('close') and no new order is placed from that bar. Used
        by intraday strategies to be flat overnight.
      * `target_r`: profit target as a multiple of the initial risk, decided on the
        signal bar: exit when the high reaches entry + target_r * (entry - stop), at
        that price (or the open if it gaps above). The stop is checked first.
      * `size_frac`: fraction of equity to put into the position, decided on the
        signal bar; the sizer's result is capped at that (volatility targeting).

    Args:
        strategy: any `Strategy` (indicators with `want_long` and `atr`, plus
            `atr_mult`); defaults to SmaCrossStrategy().
        costs: commission/slippage model.
        sizer: position sizing rule.
        initial_capital: starting cash in dollars.
    """

    def __init__(self, strategy: Strategy | None = None, costs: CostModel | None = None,
                 sizer: PositionSizer | None = None, initial_capital: float = 10_000.0):
        self.strategy = strategy or SmaCrossStrategy()
        self.costs = costs or CostModel()
        self.sizer = sizer or PositionSizer()
        self.initial_capital = initial_capital

    def run(self, bars: pd.DataFrame, start=None) -> BacktestResult:
        """Backtest `bars` (indexed OHLCV, see BarLoader) and return the result.

        If `start` is given, indicators are computed on all `bars` (so earlier bars
        serve as warm-up) but trading and the equity curve begin at `start`.
        """
        if bars.empty:
            raise ValueError("no bars to backtest")
        df = self.strategy.indicators(bars)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
            if df.empty:
                raise ValueError(f"no bars on or after start {start}")
        idx = df.index
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = df["atr"].to_numpy(float)
        want = df["want_long"].to_numpy(bool)
        n = len(df)
        stop_px = df["stop_price"].to_numpy(float) if "stop_price" in df else np.full(n, np.nan)
        eod = df["exit_at_close"].to_numpy(bool) if "exit_at_close" in df else np.zeros(n, bool)
        frac = df["size_frac"].to_numpy(float) if "size_frac" in df else np.full(n, np.nan)
        tgt_r = df["target_r"].to_numpy(float) if "target_r" in df else np.full(n, np.nan)

        cash = self.initial_capital
        shares = 0
        stop = entry_fill = entry_comm = 0.0
        target = np.inf
        entry_time = None
        pending = None                      # 'buy' / 'sell' decided at prior close
        pending_atr = pending_stop = pending_frac = pending_tgt = np.nan
        equity = np.empty(n)
        held = np.zeros(n, dtype=int)
        trades = []

        def close_position(i, price, reason):
            nonlocal cash, shares
            fill = self.costs.sell_price(price)
            comm = self.costs.commission(shares, fill, sell=True)
            cash += shares * fill - comm
            trades.append(Trade(entry_time, idx[i], shares, entry_fill, fill,
                                entry_comm + comm, reason))
            shares = 0

        for i in range(n):
            # 1. Fill the order decided at the previous close, at this open.
            if pending == "buy" and shares == 0:
                fill = self.costs.buy_price(o[i])
                new_stop = (pending_stop if not np.isnan(pending_stop)
                            else fill - self.strategy.atr_mult * pending_atr)
                qty = self.sizer.shares(cash, fill, fill - new_stop)
                if not np.isnan(pending_frac):
                    qty = min(qty, int(cash * max(pending_frac, 0.0) // fill))
                while qty > 0 and qty * fill + self.costs.commission(qty, fill) > cash:
                    qty -= 1
                if qty > 0:
                    entry_comm = self.costs.commission(qty, fill)
                    cash -= qty * fill + entry_comm
                    shares, entry_fill, entry_time = qty, fill, idx[i]
                    stop = new_stop
                    target = (fill + pending_tgt * (fill - new_stop)
                              if np.isfinite(pending_tgt) and fill > new_stop else np.inf)
            elif pending == "sell" and shares > 0:
                close_position(i, o[i], "signal")
            pending = None

            # 2. Protective stop, checked against the bar's low.
            if shares > 0 and l[i] <= stop:
                close_position(i, min(o[i], stop), "stop")
            # 2b. Profit target (after the stop: if a bar touches both, the stop counts).
            if shares > 0 and h[i] >= target:
                close_position(i, max(o[i], target), "target")

            # 3. Intraday strategies: flat at this bar's close.
            if eod[i] and shares > 0:
                close_position(i, c[i], "close")

            # 4. Mark to market, then 5. decide the next order from this close.
            equity[i] = cash + shares * c[i]
            held[i] = shares
            if i + 1 < n and not eod[i]:
                if shares > 0 and not want[i]:
                    pending = "sell"
                elif shares == 0 and want[i] and (i == 0 or not want[i - 1]) \
                        and not (np.isnan(atr[i]) and np.isnan(stop_px[i])):
                    pending, pending_atr, pending_stop = "buy", atr[i], stop_px[i]
                    pending_frac = frac[i]
                    pending_tgt = tgt_r[i]

        if shares > 0:
            close_position(n - 1, c[-1], "end")
            equity[-1] = cash
            held[-1] = 0

        return BacktestResult(self.initial_capital, pd.Series(equity, index=idx), trades,
                              pd.Series(held, index=idx))
