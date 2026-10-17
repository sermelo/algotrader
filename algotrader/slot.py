"""Small-account intraday trading: a few positions at a time, the most promising stocks.

With 1,000 to 2,000 euros there is room for one to three positions (`slots`, each a
`1 / slots` share of the account), so the strategy has to choose: at each decision minute a
model gives every stock an `edge` (its expected return over the next `hold_minutes`, before
costs); the engine buys the stocks with the highest edges that also clear their own cost
hurdle (`breakeven`) while a slot is free (optionally only stocks inside a price window, below a maximum round-trip cost, and invested to at least a minimum share of the slot), and sells each after `hold_minutes` or at the
close of the session, whichever comes first. At most `max_trades_per_day` trades a day
(default: one per slot: in a cash account the proceeds of a sale only settle the next day,
T+1, and buying again with unsettled money before then is a free-riding violation, so each
slot's money can make one round trip a day); a stock is bought at most once a day.

Execution is the same as in the other engines, without look-ahead: the edge for minute t
is known at that bar's close and the fill is the next bar's open (plus the half-spread);
the exit is the open of the first bar at or after entry + `hold_minutes`, or the last bar's
close if the session ends first. Whole shares, `CostModel` commissions and fees, the whole
account in one stock, profits compounded.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .stats import returns_stats


def breakeven(cost: CostModel, price: float, value: float, extra_bps: float = 0.0) -> float:
    """Round-trip cost as a fraction of the position: how far the price must rise from the
    buy to the sell just to cover commissions, fees and spread on a buy of about `value` at
    `price` (whole shares; NaN if not even one share fits). `extra_bps` adds other costs
    per round trip, e.g. 0.03% (3 bp) per currency conversion."""
    shares = int(value // price)
    if shares < 1 or price <= 0:
        return float("nan")
    pos = shares * price
    spread = shares * (cost.buy_price(price) - price) + shares * (price - cost.sell_price(price))
    fees = cost.commission(shares, price) + cost.commission(shares, price, sell=True)
    return (fees + spread) / pos + extra_bps / 1e4


@dataclass
class SlotResult:
    trades: pd.DataFrame
    equity: pd.Series                       # account value at the end of each day
    capital: float

    def summary(self) -> dict:
        r = self.equity.pct_change()
        r.iloc[0] = self.equity.iloc[0] / self.capital - 1
        st = returns_stats(r)
        t = self.trades
        return {"trades": len(t), "days": len(self.equity),
                "win_rate": float((t["net"] > 0).mean()) if len(t) else float("nan"),
                "mean_net_pct": float(t["net_pct"].mean()) if len(t) else float("nan"),
                "mean_cost_pct": float(t["cost_pct"].mean()) if len(t) else float("nan"),
                "total_return": st["return"], "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"]}


@dataclass(frozen=True)
class SlotTrader:
    capital: float = 1000.0
    slots: int = 1
    cost: CostModel = field(default_factory=CostModel)
    hold_minutes: int = 90
    min_margin: float = 0.0                 # the edge must beat the hurdle by this fraction
    max_trades_per_day: int = None          # default: one per slot
    extra_bps: float = 0.0                  # other costs per round trip (e.g. currency conversion)
    min_price: float = 0.0                  # price filters, applied to the decision bar's close
    max_price: float = float("inf")
    max_cost: float = None                  # reject a stock whose round-trip cost exceeds this fraction
    min_invested: float = 0.0               # whole shares must cover at least this share of the slot

    def __post_init__(self):
        if self.min_price < 0 or self.max_price <= self.min_price or not 0 <= self.min_invested <= 1 \
                or (self.max_cost is not None and self.max_cost <= 0):
            raise ValueError("require 0 <= min_price < max_price, 0 <= min_invested <= 1, max_cost > 0")
        if (self.capital <= 0 or self.slots < 1 or self.hold_minutes < 1 or self.min_margin < 0
                or (self.max_trades_per_day is not None and self.max_trades_per_day < 1)):
            raise ValueError("require capital > 0, slots >= 1, hold_minutes >= 1, max_trades_per_day >= 1, min_margin >= 0")

    def run(self, bars: dict, edges: pd.DataFrame) -> SlotResult:
        """`bars`: ticker -> minute bars (open, high, low, close; index = bar start, one
        session a day); `edges`: (decision minute x ticker) expected return over the holding
        period, known at that bar's close (NaN = no view)."""
        equity, rows, daily = self.capital, [], {}
        hold = pd.Timedelta(minutes=self.hold_minutes)
        limit = self.max_trades_per_day or self.slots
        for day, block in edges.groupby(edges.index.normalize()):
            value, today = equity / self.slots, []           # each slot is funded from the start-of-day equity
            for ts, row in block.iterrows():
                row = row.dropna()
                while len(today) < limit and not row.empty:
                    held = {r["ticker"] for r in today}
                    trade = self._try(bars, ts, row[~row.index.isin(held)], value, hold, today)
                    if trade is None:
                        break
                    today.append(trade)
                if len(today) >= limit:
                    break
            equity += sum(r["net"] for r in today)
            rows.extend(today)
            daily[day] = equity
        trades = pd.DataFrame(rows)
        return SlotResult(trades, pd.Series(daily, dtype=float), self.capital)

    def _try(self, bars, ts, row, equity, hold, today):
        for ticker, edge in row.sort_values(ascending=False).items():
            b = bars.get(ticker)
            if b is None or ts not in b.index:
                continue
            i = b.index.get_loc(ts)
            if i + 1 >= len(b) or b.index[i + 1].normalize() != ts.normalize():
                continue
            price = float(b["close"].iloc[i])
            if not self.min_price <= price <= self.max_price:
                continue
            hurdle = breakeven(self.cost, price, equity, self.extra_bps)
            if not np.isfinite(hurdle) or edge < hurdle + self.min_margin:
                continue
            if self.max_cost is not None and hurdle > self.max_cost:
                continue
            if int(equity // price) * price < self.min_invested * equity:
                continue
            entry = b.index[i + 1]
            if sum(1 for r in today if r["entry"] <= entry < r["exit"]) >= self.slots:
                return None                                        # every slot is busy at that moment
            open_px = float(b["open"].iloc[i + 1])
            fill = self.cost.buy_price(open_px)
            shares = int(equity // fill)
            while shares > 0 and shares * fill + self.cost.commission(shares, fill) > equity:
                shares -= 1
            if shares < 1:
                continue
            session = b[b.index.normalize() == ts.normalize()]
            later = session[session.index >= entry + hold]
            if len(later):
                exit_time, px, reason = later.index[0], float(later["open"].iloc[0]), "horizon"
            else:
                exit_time, px, reason = session.index[-1], float(session["close"].iloc[-1]), "close"
            sell = self.cost.sell_price(px)
            buy_comm = self.cost.commission(shares, fill)
            sell_comm = self.cost.commission(shares, sell, sell=True)
            net = shares * (sell - fill) - buy_comm - sell_comm
            return {"ticker": ticker, "decision": ts, "entry": entry, "exit": exit_time, "shares": shares,
                    "buy": fill, "sell": sell, "edge": float(edge), "hurdle": float(hurdle),
                    "commission": buy_comm + sell_comm, "net": net, "net_pct": net / (shares * fill),
                    "cost_pct": (buy_comm + sell_comm + shares * (fill - open_px) + shares * (px - sell)) / (shares * fill),
                    "reason": reason}
        return None
