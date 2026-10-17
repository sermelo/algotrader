"""How far do prices move over a few minutes to a few hours, compared with what a trade costs?

For the small-account intraday plan a trade needs a minimum move to cover its break-even (about 0.08% to
0.2%, see `slot.breakeven`). This measures, on the raw minute bars of S&P 500 members, the distribution of the
price change from the last trade at minute t to the last trade at minute t + h (h = 1, 5, 10 ... 240) for every
member and every start minute between 10:00 and 15:00 whose window ends inside the session: mean, spread, tails
against a normal distribution with the same spread, the share of windows that rise or fall by more than each
threshold, and a histogram. Windows overlap (every minute is a start); the numbers describe the distribution,
not independent trades.
"""
from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np
import pandas as pd

from .minutefeat import session_arrays

HORIZONS = (1, 2, 5, 10, 15, 20, 30, 60, 90, 120, 240)
THRESHOLDS = (0.0008, 0.0014, 0.0019)         # break-even of one 2,000-euro position, a 667-euro slot, a 500-euro slot
EDGES = np.linspace(-0.05, 0.05, 1001)         # 0.01% bins, the outer bins hold everything beyond +-5%


def normal_tail(c: float, sd: float) -> float:
    """P(|X| > c) for a normal variable with mean 0 and standard deviation sd."""
    return 1.0 - erf(c / (sd * sqrt(2))) if sd > 0 else 0.0


@dataclass
class Accumulator:
    """Running moments, threshold counts and a histogram of price changes (fractions) for one horizon."""
    thresholds: tuple = THRESHOLDS
    n: int = 0
    zero: int = 0
    abs_sum: float = 0.0
    s: np.ndarray = field(default_factory=lambda: np.zeros(5))             # sums of x^0 ... x^4
    up: np.ndarray = None
    down: np.ndarray = None
    hist: np.ndarray = field(default_factory=lambda: np.zeros(len(EDGES) - 1, dtype=np.int64))

    def __post_init__(self):
        self.up = np.zeros(len(self.thresholds), dtype=np.int64)
        self.down = np.zeros(len(self.thresholds), dtype=np.int64)

    def add(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if not len(x):
            return
        self.n += len(x)
        self.zero += int((x == 0).sum())
        self.abs_sum += float(np.abs(x).sum())
        for k in range(1, 5):
            self.s[k] += (x ** k).sum()
        for i, c in enumerate(self.thresholds):
            self.up[i] += int((x > c).sum())
            self.down[i] += int((x < -c).sum())
        self.hist += np.histogram(np.clip(x, EDGES[0], EDGES[-1] - 1e-12), bins=EDGES)[0]

    def merge(self, other: "Accumulator"):
        self.n += other.n
        self.zero += other.zero
        self.abs_sum += other.abs_sum
        self.s += other.s
        self.up += other.up
        self.down += other.down
        self.hist += other.hist

    def summary(self) -> dict:
        n = max(self.n, 1)
        m = self.s[1] / n
        var = self.s[2] / n - m * m
        sd = sqrt(max(var, 0.0))
        m4 = self.s[4] / n - 4 * m * self.s[3] / n + 6 * m * m * self.s[2] / n - 3 * m ** 4
        cum = np.cumsum(self.hist) / max(self.hist.sum(), 1)
        q = lambda p: float(np.interp(p, cum, EDGES[1:]))              # quantile from the 0.01% histogram
        robust = (q(0.75) - q(0.25)) / 1.349                           # the sd a normal distribution with this IQR would have
        out = {"n": self.n, "mean": m, "sd": sd, "mean_abs": self.abs_sum / n, "robust_sd": robust, "zero_share": self.zero / n,
               "excess_kurtosis": m4 / var ** 2 - 3 if var > 0 else float("nan"), "tails": []}
        for i, c in enumerate(self.thresholds):
            out["tails"].append({"threshold": c, "up": self.up[i] / n, "down": self.down[i] / n,
                                 "either": (self.up[i] + self.down[i]) / n, "normal_either": normal_tail(c, sd)})
        return out


def window_changes(df: pd.DataFrame, members: pd.DataFrame, horizons=HORIZONS, first_start: int = 30,
                   last_start: int = 330, min_price: float = 0.0):
    """Yield (h, changes): fractional change of the last trade price from minute t to t + h for every member and
    session, t from `first_start` to `last_start` (minutes after 09:30), the window ending inside the session
    (a 13:00 close shortens it), the stock having traded by minute t and costing at least `min_price` there."""
    days, tickers, A, vol, has, end, close_f, seen = session_arrays(df)
    member = members.reindex(index=days, columns=tickers, fill_value=False).to_numpy(bool)
    ts = np.arange(first_start, last_start + 1)
    c0 = close_f[:, :, ts]
    ok0 = seen[:, :, ts] & member[:, :, None] & (c0 >= min_price) & (c0 > 0)
    for h in horizons:
        ok = ok0 & ((ts + h)[None, None, :] < end[:, None, None])
        ok &= np.isfinite(close_f[:, :, np.minimum(ts + h, close_f.shape[-1] - 1)])
        c1 = close_f[:, :, np.minimum(ts + h, close_f.shape[-1] - 1)]
        yield h, (c1 / c0 - 1)[ok]
