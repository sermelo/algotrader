"""Performance statistics."""
import numpy as np
import pandas as pd

from .engine import BacktestResult

TRADING_DAYS = 252


class Metrics:
    """Summary statistics for a BacktestResult.

    Sharpe is computed on DAILY equity returns (last equity of each calendar day)
    and annualised with sqrt(252); risk-free rate is taken as 0.

    Args:
        result: the backtest to summarise.
        bars: optional bars, used only for the buy-and-hold benchmark.
    """

    def __init__(self, result: BacktestResult, bars: pd.DataFrame = None):
        self.result = result
        self.bars = bars

    @property
    def daily_equity(self) -> pd.Series:
        return self.result.equity.resample("1D").last().dropna()

    def total_return(self) -> float:
        return self.result.equity.iloc[-1] / self.result.initial_capital - 1

    def cagr(self) -> float:
        """Compound annual growth rate; NaN if under a day of data or equity <= 0."""
        eq = self.result.equity
        years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400)
        growth = 1 + self.total_return()
        return growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")

    @staticmethod
    def _sharpe(daily_values: pd.Series) -> float:
        r = daily_values.pct_change().dropna()
        if len(r) < 2 or r.std() == 0:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))

    def sharpe(self) -> float:
        """Annualised Sharpe from daily returns; NaN if undefined."""
        return self._sharpe(self.daily_equity)

    def buy_and_hold_sharpe(self) -> float:
        """Sharpe of holding the bars: first open, then each day's last close (NaN w/o bars)."""
        if self.bars is None or self.bars.empty:
            return float("nan")
        daily = self.bars["close"].resample("1D").last().dropna()
        start = pd.Series([self.bars["open"].iloc[0]], index=[daily.index[0] - pd.Timedelta("1D")])
        return self._sharpe(pd.concat([start, daily]))

    def max_drawdown(self) -> float:
        """Largest peak-to-trough fall of bar-level equity (a negative number or 0)."""
        eq = self.result.equity
        return float((eq / eq.cummax() - 1).min())

    def buy_and_hold(self) -> float:
        """Return of holding from the first open to the last close (NaN w/o bars)."""
        if self.bars is None or self.bars.empty:
            return float("nan")
        return float(self.bars["close"].iloc[-1] / self.bars["open"].iloc[0] - 1)

    def exposure(self) -> float:
        """Fraction of bars that end with a position open (NaN if not recorded)."""
        pos = self.result.position
        if pos is None or pos.empty:
            return float("nan")
        return float((pos > 0).mean())

    def trade_stats(self) -> dict:
        """Trade counts and P&L statistics. Break-even trades count as losses.

        Profit factor is gross profit / gross loss: inf with wins and no losses,
        NaN with no trades or no wins and no losses.
        """
        pnl = np.array([t.pnl for t in self.result.trades], dtype=float)
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        gross_loss = -losses.sum()
        return {
            "trades": len(pnl),
            "win_rate": len(wins) / len(pnl) if len(pnl) else float("nan"),
            "avg_win": wins.mean() if len(wins) else 0.0,
            "avg_loss": losses.mean() if len(losses) else 0.0,
            "profit_factor": (wins.sum() / gross_loss if gross_loss > 0
                              else float("inf") if len(wins) else float("nan")),
            "total_commission": sum(t.commission for t in self.result.trades),
        }

    def summary(self) -> dict:
        """All statistics as a flat dict."""
        return {"total_return": self.total_return(), "cagr": self.cagr(),
                "sharpe": self.sharpe(), "max_drawdown": self.max_drawdown(),
                "buy_and_hold": self.buy_and_hold(),
                "buy_and_hold_sharpe": self.buy_and_hold_sharpe(), "exposure": self.exposure(),
                "final_equity": float(self.result.equity.iloc[-1]),
                **self.trade_stats()}
