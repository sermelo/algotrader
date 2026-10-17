"""Multi-asset portfolio backtests for cross-sectional strategies on daily bars.

A cross-sectional strategy looks at all tickers at once and returns target weights
(fractions of equity; negative = short when the backtester allows it) for each day,
decided from closes up to and including that day. The engine trades to those weights at the NEXT
day's open, using the same CostModel as the single-ticker engine, and marks the
portfolio to each day's close.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class PortfolioResult:
    """Output of a portfolio backtest."""
    initial_capital: float
    equity: pd.Series                         # at each day's close
    holdings: pd.DataFrame                    # shares per ticker at each close
    orders: list = field(default_factory=list)   # (date, ticker, shares, fill price, commission)

    @property
    def rebalances(self) -> int:
        """Days on which at least one order was filled."""
        return len({o[0] for o in self.orders})

    @property
    def turnover(self) -> float:
        """Traded value over average equity, per year."""
        if not self.orders or self.equity.empty:
            return 0.0
        traded = sum(abs(o[2]) * o[3] for o in self.orders)
        years = max((self.equity.index[-1] - self.equity.index[0]).days / 365.25, 1 / 365.25)
        return traded / self.equity.mean() / years


def panel(frames: dict, field_name: str) -> pd.DataFrame:
    """Dates x tickers table of one OHLCV field (NaN where a ticker has no bar)."""
    return pd.DataFrame({t: f[field_name] for t, f in frames.items()}).sort_index()


class PortfolioBacktester:
    """Rebalances a portfolio to a strategy's target weights.

    Rules:
      * weights decided on day t's close are traded at day t+1's open (each ticker at
        its own open; a ticker without a bar that day is traded at its next bar);
      * orders are placed only when the strategy's weights change, so holdings drift
        between rebalances instead of being traded back every day;
      * shares are whole; sells are filled before buys, and buys are scaled down if
        cash is short;
      * a ticker whose data ends is closed out at its last close (delisting/rename);
      * equity = cash + shares * last known close (short shares are negative; short
        sale proceeds are added to cash);
      * long only unless `allow_short`; then weights may be negative, gross exposure
        (sum of |weights|) is scaled down to at most `max_gross`, and short positions
        pay `borrow_bps` a year on their market value, charged daily (1/252).
    """

    def __init__(self, strategy, costs: CostModel | None = None,
                 initial_capital: float = 10_000.0, allow_short: bool = False,
                 max_gross: float = 1.0, borrow_bps: float = 0.0):
        if not max_gross > 0 or borrow_bps < 0:
            raise ValueError("require max_gross > 0 and borrow_bps >= 0")
        self.strategy = strategy
        self.costs = costs or CostModel()
        self.initial_capital = initial_capital
        self.allow_short, self.max_gross, self.borrow_bps = allow_short, max_gross, borrow_bps

    def run(self, frames: dict, start=None) -> PortfolioResult:
        """Backtest daily `frames` ({ticker: bars}); trading starts at `start`."""
        if not frames:
            raise ValueError("no tickers")
        opens, closes = panel(frames, "open"), panel(frames, "close")
        weights = self.strategy.weights(closes).reindex(index=closes.index,
                                                        columns=closes.columns).fillna(0.0)
        if start is not None:
            keep = closes.index >= pd.Timestamp(start)
            opens, closes, weights = opens[keep], closes[keep], weights[keep]
            if closes.empty:
                raise ValueError(f"no bars on or after start {start}")
        last_bar = {t: f.index[-1] for t, f in frames.items()}
        tickers = list(closes.columns)
        shares = pd.Series(0, index=tickers, dtype=np.int64)
        last_close = pd.Series(np.nan, index=tickers)
        cash = self.initial_capital
        target, pending, prev_w = None, set(), None
        equity, held, orders = [], [], []

        for day in closes.index:
            o, c = opens.loc[day], closes.loc[day]
            # 1. trade the pending tickers towards the latest targets at today's open
            if pending:
                value = cash + float((shares * last_close.fillna(0)).sum())
                ready = [t for t in pending if not np.isnan(o[t])]
                # whole shares, rounded towards zero (also for shorts)
                want = {t: int(value * target[t] / o[t]) for t in ready}
                for t in [t for t in ready if want[t] < shares[t]]:          # sells first
                    q = int(want[t] - shares[t])
                    fill = self.costs.sell_price(o[t])
                    comm = self.costs.commission(-q, fill, sell=True)
                    cash += -q * fill - comm
                    shares[t] += q
                    orders.append((day, t, q, fill, comm))
                for t in [t for t in ready if want[t] > shares[t]]:
                    q = int(want[t] - shares[t])
                    fill = self.costs.buy_price(o[t])
                    while q > 0 and q * fill + self.costs.commission(q, fill) > cash:
                        q = int(min(q - 1, (cash - self.costs.commission(q, fill)) // fill))
                    if q > 0:
                        comm = self.costs.commission(q, fill)
                        cash -= q * fill + comm
                        shares[t] += q
                        orders.append((day, t, q, fill, comm))
                pending -= set(ready)
            last_close = c.where(c.notna(), last_close)
            # 2. liquidate tickers whose data ends today
            for t in tickers:
                if shares[t] and last_bar[t] == day:
                    q = int(shares[t])
                    fill = self.costs.sell_price(c[t]) if q > 0 else self.costs.buy_price(c[t])
                    comm = self.costs.commission(abs(q), fill, sell=q > 0)
                    cash += q * fill - comm
                    orders.append((day, t, -int(shares[t]), fill, comm))
                    shares[t] = 0
                    pending.discard(t)
            # 3. borrow fee on shorts, then mark to market
            mv = shares * last_close.fillna(0)
            if self.borrow_bps:
                cash -= float(-mv[mv < 0].sum()) * self.borrow_bps / 1e4 / 252
            equity.append(cash + float(mv.sum()))
            held.append(shares.copy())
            # 4. new targets only when the strategy's weights change (no drift trading)
            w = weights.loc[day]
            if prev_w is None or float((w - prev_w).abs().max()) > 1e-12:
                prev_w = w
                alive = pd.Series({t: last_bar[t] > day for t in tickers})
                target = w.where(alive, 0.0)
                if not self.allow_short:
                    target = target.clip(lower=0)
                limit = self.max_gross if self.allow_short else 1.0
                gross = float(target.abs().sum())
                if gross > limit:
                    target = target * (limit / gross)
                pending = {t for t in tickers if target[t] != 0 or shares[t] != 0}

        idx = closes.index
        return PortfolioResult(self.initial_capital, pd.Series(equity, index=idx),
                               pd.DataFrame(held, index=idx), orders)


def _rebalance_days(index: pd.DatetimeIndex, every: str) -> np.ndarray:
    """True on the last trading day of each period ('W' weeks, 'M' months, 'D' daily)."""
    if every == "D":
        return np.ones(len(index), dtype=bool)
    period = pd.Series(index.to_period(every), index=index)
    return (period != period.shift(-1)).to_numpy()


@dataclass(frozen=True)
class CrossSectionalMomentum:
    """Jegadeesh & Titman (1993), long side: hold the `top_n` tickers with the highest
    return from `lookback` to `skip` days ago, equal weight, rebalanced at each
    period end (`every`). Between rebalances the weights are held."""
    lookback: int = 252
    skip: int = 21
    top_n: int = 10
    every: str = "M"
    short_n: int = 0              # > 0: also short the bottom short_n (long-short)

    def __post_init__(self):
        if self.lookback < 2 or not 0 <= self.skip < self.lookback or self.top_n < 1:
            raise ValueError("require lookback >= 2, 0 <= skip < lookback, top_n >= 1")
        if self.short_n < 0:
            raise ValueError("short_n must be >= 0")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        return closes.ffill().shift(self.skip) / closes.ffill().shift(self.lookback) - 1

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        score = self.score(closes)
        w = _top_weights(score, closes, self.top_n, self.every, largest=True)
        if self.short_n:
            w = w - _top_weights(score, closes, self.short_n, self.every, largest=False)
        return w


@dataclass(frozen=True)
class ShortTermReversal:
    """Jegadeesh (1990) / Lehmann (1990), long side: hold the `top_n` tickers with the
    LOWEST return over the last `lookback` days, rebalanced at each period end."""
    lookback: int = 5
    top_n: int = 10
    every: str = "W"

    def __post_init__(self):
        if self.lookback < 1 or self.top_n < 1:
            raise ValueError("require lookback >= 1, top_n >= 1")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        return closes.ffill() / closes.ffill().shift(self.lookback) - 1

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return _top_weights(self.score(closes), closes, self.top_n, self.every, largest=False)


def _top_weights(score, closes, top_n, every, largest) -> pd.DataFrame:
    """Equal weights on the top_n scores (tickers trading that day only) at each
    rebalance day, held (forward-filled) until the next."""
    rebal = _rebalance_days(closes.index, every)
    tradable = score.where(closes.notna())
    ranks = tradable.rank(axis=1, ascending=not largest, method="first")
    chosen = (ranks <= top_n) & tradable.notna()
    w = chosen.div(chosen.sum(axis=1).clip(lower=1), axis=0).astype(float)
    w[~rebal] = np.nan
    w.iloc[0] = w.iloc[0].fillna(0.0)
    return w.ffill().fillna(0.0)


@dataclass(frozen=True)
class PairsStrategy:
    """Mean reversion of the log price ratio of fixed pairs (needs shorting).

    For each pair (a, b): spread = log(a) - log(b), z = (spread - mean) / std over
    the last `lookback` days. Enter when |z| > `entry_z` (long the cheap leg, short
    the rich one), exit when |z| < `exit_z`. Each open pair holds +-0.5 / n_pairs per
    leg. Pairs are chosen in advance on economic grounds, not by searching the data.
    """
    pairs: tuple = (("XOM", "CVX"), ("JPM", "BAC"), ("KO", "PEP"), ("SPY", "DIA"),
                    ("QQQ", "XLK"), ("AAPL", "MSFT"))
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5

    def __post_init__(self):
        if self.lookback < 2 or not 0 <= self.exit_z < self.entry_z:
            raise ValueError("require lookback >= 2 and 0 <= exit_z < entry_z")

    def zscores(self, closes: pd.DataFrame) -> pd.DataFrame:
        filled = closes.ffill()
        out = {}
        for a, b in self.pairs:
            if a in closes and b in closes:
                spread = np.log(filled[a]) - np.log(filled[b])
                mean = spread.rolling(self.lookback).mean()
                std = spread.rolling(self.lookback).std()
                out[f"{a}/{b}"] = ((spread - mean) / std).where(closes[a].notna() & closes[b].notna())
        return pd.DataFrame(out, index=closes.index)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        z = self.zscores(closes)
        w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        live = [p for p in self.pairs if f"{p[0]}/{p[1]}" in z]
        if not live:
            return w
        leg = 0.5 / len(live)
        for a, b in live:
            zs = z[f"{a}/{b}"].to_numpy()
            state = np.zeros(len(zs))
            pos = 0
            for i, v in enumerate(zs):
                if np.isnan(v):
                    pos = 0
                elif pos == 0 and v > self.entry_z:
                    pos = -1                       # a rich: short a, long b
                elif pos == 0 and v < -self.entry_z:
                    pos = 1
                elif pos != 0 and abs(v) < self.exit_z:
                    pos = 0
                state[i] = pos
            w[a] += state * leg
            w[b] -= state * leg
        return w


def xs_features(closes: pd.DataFrame) -> dict:
    """Daily features per ticker, each ranked across tickers per day (0..1).

    Returns {name: dates x tickers}. All use closes up to that day only.
    """
    c = closes.ffill()
    r1 = c.pct_change(fill_method=None)
    feats = {f"ret_{n}": c / c.shift(n) - 1 for n in (1, 5, 21, 63, 126)}
    feats["vol_21"] = r1.rolling(21).std()
    feats["dist_sma50"] = c / c.rolling(50).mean() - 1
    feats["dist_sma200"] = c / c.rolling(200).mean() - 1
    gain = r1.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-r1.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    feats["rsi_14"] = 100 - 100 / (1 + gain / loss)
    live = closes.notna()
    return {k: f.where(live).replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
            for k, f in feats.items()}


@dataclass
class CrossSectionalML:
    """Gradient boosting ranks tickers by expected forward return; hold the top N.

    Label: the ticker's `horizon`-day forward return minus that day's cross-sectional
    mean (only rows whose future lies inside the training frames are used). Fit with
    `fit_many(frames)` on data before the test period; `weights(closes)` then scores
    every day and, at each period end (`every`), holds the `top_n` best equally
    (and shorts the `short_n` worst when `short_n` > 0).
    """
    horizon: int = 5
    top_n: int = 10
    short_n: int = 0
    every: str = "W"
    max_iter: int = 200
    random_state: int = 0
    model: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.horizon < 1 or self.top_n < 1 or self.short_n < 0:
            raise ValueError("require horizon >= 1, top_n >= 1, short_n >= 0")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    @staticmethod
    def _stack(feats: dict) -> pd.DataFrame:
        return pd.DataFrame({k: f.stack(future_stack=True) for k, f in feats.items()})

    def fit_many(self, frames) -> "CrossSectionalML":
        from sklearn.ensemble import HistGradientBoostingRegressor

        frames = dict(frames) if isinstance(frames, dict) else dict(enumerate(frames))
        closes = panel(frames, "close")
        X = self._stack(xs_features(closes))
        fwd = closes.ffill().shift(-self.horizon) / closes.ffill() - 1
        fwd = fwd.where(closes.notna() & closes.shift(-self.horizon).notna())
        y = fwd.sub(fwd.mean(axis=1), axis=0).stack(future_stack=True).reindex(X.index)
        ok = y.notna() & X.notna().any(axis=1)
        if ok.sum() < 50:
            raise ValueError("not enough training rows")
        self.model = HistGradientBoostingRegressor(max_depth=3, max_iter=self.max_iter,
                                                   learning_rate=0.05, l2_regularization=1.0,
                                                   random_state=self.random_state)
        self.model.fit(X[ok].to_numpy(), y[ok].to_numpy())
        return self

    def score(self, closes: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit_many() before weights()")
        X = self._stack(xs_features(closes))
        pred = pd.Series(self.model.predict(X.to_numpy()), index=X.index)
        return pred.unstack().reindex(index=closes.index, columns=closes.columns)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        s = self.score(closes)
        w = _top_weights(s, closes, self.top_n, self.every, largest=True)
        if self.short_n:
            w = w - _top_weights(s, closes, self.short_n, self.every, largest=False)
        return w
