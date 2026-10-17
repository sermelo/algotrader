"""Price-chart CNN (roadmap S9).

Jiang, Kelly & Xiu, "(Re-)Imag(in)ing Price Trends" (Journal of Finance 2023): a
convolutional network reads a picture of the last `d` days (an OHLC bar chart with a
moving-average line and volume bars) and predicts whether the stock's return over the
next `h` days is positive; stocks are sorted on that probability.

Image (as in the paper, black and white, 3 pixels a day): the left pixel of each day's
column triple marks the open, the middle column holds the high-low line, the right pixel
marks the close; a moving average of the closes over `d` days is one pixel per day; the
bottom fifth shows volume bars. Prices are divided by the window's first close and
scaled to the window's own min and max, volume to the window's maximum, so the picture
has no scale and no level, only shape. The moving average uses closes from before the
window (2d - 1 days in all).

Deviations from the paper (CPU only, no GPU): channels (16, 32, 64) instead of
(64, 128, 256); learning rate 1e-3 and at most `epochs` epochs with early stopping on a
later validation block instead of lr 1e-5 and 100 epochs; a rolling window of `train_years`
years sampled every `stride` days instead of 1993-2000 on every stock-day; a single model,
not an ensemble of five; trained on the S&P 500 members of each day only (the paper uses
all US stocks). Scores are only computed on rebalance days (month ends).

No look-ahead: the model for year Y is fitted on images whose label (the return over the
next `h` days) was known before 1 January Y, and every score uses data up to its own day.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import _rebalance_days, _top_weights

VOLUME_ROWS = 12                     # of 64 rows for a 20-day chart (price 51, gap 1, volume 12)


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------- images

def image_height(d: int) -> int:
    """Rows of the chart for `d` days: 32, 64, 96 for 5, 20, 60 days (as in the paper)."""
    return {5: 32, 20: 64, 60: 96}.get(d, 3 * d + 4)


def windows(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """(N, n) values of `arr` over the n rows ending at each (row, col) pair."""
    return arr[rows[:, None] + np.arange(-(n - 1), 1)[None, :], cols[:, None]]


def render_charts(o, h, l, c, v, ma, height: int = None) -> np.ndarray:
    """Charts (N, 1, height, 3d) of uint8 0/255 from (N, d) arrays of open, high, low,
    close, volume and moving average."""
    N, d = c.shape
    height = height or image_height(d)
    vol_rows = max(2, round(height * 12 / 64))
    price_rows = height - vol_rows - 1
    base = c[:, :1]
    o, h, l, c, ma = (x / base for x in (o, h, l, c, ma))
    top = np.maximum(h.max(axis=1), ma.max(axis=1))[:, None]
    bot = np.minimum(l.min(axis=1), ma.min(axis=1))[:, None]
    span = np.where(top > bot, top - bot, 1.0)

    def row(x):
        return np.clip(np.rint((top - x) / span * (price_rows - 1)), 0, price_rows - 1).astype(int)
    ro, rh, rl, rc, rm = row(o), row(h), row(l), row(c), row(ma)
    img = np.zeros((N, height, 3 * d), dtype=np.uint8)
    rr = np.arange(price_rows)[None, :, None]
    img[:, :price_rows, 1::3] = (rr >= rh[:, None, :]) & (rr <= rl[:, None, :])     # high-low line
    n_idx = np.repeat(np.arange(N), d)
    d_idx = np.tile(np.arange(d), N)
    img[n_idx, ro.ravel(), 3 * d_idx] = 1                                         # open, left
    img[n_idx, rc.ravel(), 3 * d_idx + 2] = 1                                     # close, right
    for k in range(3):
        img[n_idx, rm.ravel(), 3 * d_idx + k] = 1                                 # moving average
    vmax = np.where(v.max(axis=1) > 0, v.max(axis=1), 1.0)[:, None]
    vh = np.rint(v / vmax * vol_rows).astype(int)                                 # bar heights
    rv = np.arange(vol_rows)[None, :, None]
    img[:, height - vol_rows:, 1::3] |= (rv >= (vol_rows - vh)[:, None, :]) & (vh[:, None, :] > 0)
    return (img * 255)[:, None]


def chart_batch(panels: dict, rows: np.ndarray, cols: np.ndarray, d: int, height: int = None) -> np.ndarray:
    """Charts for the windows ending at (rows, cols) of the (dates x tickers) panels."""
    c_ext = windows(panels["close"], rows, cols, 2 * d - 1)
    ma = np.stack([c_ext[:, i:i + d].mean(axis=1) for i in range(d)], axis=1)
    g = lambda k: windows(panels[k], rows, cols, d)
    return render_charts(g("open"), g("high"), g("low"), c_ext[:, d - 1:], g("volume"), ma, height)


# ---------------------------------------------------------------- model

def build_cnn(d: int = 20, channels=(16, 32, 64), dropout: float = 0.5, seed: int = 0):
    """JKX-style CNN: three blocks of conv (5 x 3), batch norm, leaky ReLU and 2 x 1 max
    pooling, then dropout and a linear layer to one logit (probability of a positive return)."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    layers, cin = [], 1
    for cout in channels:
        layers += [nn.Conv2d(cin, cout, (5, 3), padding=(2, 1)), nn.BatchNorm2d(cout),
                   nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
        cin = cout
    height = image_height(d)
    for _ in channels:
        height //= 2
    net = nn.Sequential(*layers, nn.Flatten(), nn.Dropout(dropout),
                        nn.Linear(cin * height * 3 * d, 1))
    return net


@dataclass
class ChartCNNModel:
    d: int = 20
    h: int = 20
    channels: tuple = (16, 32, 64)
    train_years: int = 6
    stride: int = 10
    epochs: int = 6
    patience: int = 2
    batch: int = 128
    lr: float = 1e-3
    val_share: float = 0.2
    seed: int = 0
    threads: int = None
    log: list = field(default_factory=list)

    def _panels(self, frames: dict):
        cols = [t for t in frames]
        index = None
        for f in frames.values():
            index = f.index if index is None else index.union(f.index)
        panels = {k: pd.DataFrame({t: f[k] for t, f in frames.items()}).reindex(index).to_numpy(np.float32)
                  for k in ("open", "high", "low", "close", "volume")}
        return index, cols, panels

    def _valid(self, panels, members, index, cols):
        """Bool (dates x tickers): a full chart window (2d - 1 days of prices, d of volume)
        and membership on that day."""
        ok = np.ones_like(panels["close"], dtype=bool)
        for k in ("open", "high", "low", "close"):
            x = np.isfinite(panels[k]) & (panels[k] > 0)
            ok &= pd.DataFrame(x).rolling(2 * self.d - 1).min().to_numpy() > 0
        ok &= pd.DataFrame(np.isfinite(panels["volume"])).rolling(self.d).min().to_numpy() > 0
        mem = members.reindex(columns=cols, fill_value=False).astype(float)
        mem = mem.reindex(index=index.union(mem.index)).ffill().reindex(index).fillna(0.0).to_numpy() > 0.5
        return ok & mem

    def _fit(self, panels, label, ok, rows):
        """Fit on sampled `rows` (decision days); returns a model in eval mode."""
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        pairs = [(r, c) for r in rows for c in np.flatnonzero(ok[r] & np.isfinite(label[r]))]
        if len(pairs) < 50:
            return None
        pairs = np.array(pairs)
        n_val = int(len(pairs) * self.val_share)         # the last block of days validates
        tr, va = pairs[:len(pairs) - n_val], pairs[len(pairs) - n_val:]
        net = build_cnn(self.d, self.channels, seed=self.seed)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = np.inf, None, 0

        def batch(p):
            x = chart_batch(panels, p[:, 0], p[:, 1], self.d)
            return torch.from_numpy(x).float().div_(255), torch.from_numpy(label[p[:, 0], p[:, 1]]).float()

        for epoch in range(self.epochs):
            net.train()
            order = rng.permutation(len(tr))
            for i in range(0, len(order), self.batch):
                x, y = batch(tr[order[i:i + self.batch]])
                opt.zero_grad()
                loss_fn(net(x).squeeze(-1), y).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                tot = sum(loss_fn(net(x).squeeze(-1), y).item() * len(y)
                          for x, y in (batch(va[i:i + 512]) for i in range(0, len(va), 512)))
            v = tot / max(len(va), 1)
            self.log.append((epoch, v))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net

    def scores(self, frames: dict, members: pd.DataFrame, first_year: int, every: str = "M") -> pd.DataFrame:
        """Walk forward: for each year from `first_year`, fit on the earlier `train_years`
        years of images (labels known before 1 January) and score every member on each
        rebalance day of that year. Returns a (dates x tickers) panel, NaN elsewhere."""
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        index, cols, panels = self._panels(frames)
        ok = self._valid(panels, members, index, cols)
        close = panels["close"]
        label = np.full_like(close, np.nan)
        fwd = close[self.h:] / close[:-self.h] - 1
        label[:-self.h] = np.where(np.isfinite(fwd), (fwd > 0).astype(np.float32), np.nan)
        rebal = np.flatnonzero(_rebalance_days(index, every))
        out = np.full_like(close, np.nan)
        for year in range(first_year, index[-1].year + 1):
            start = int(index.searchsorted(pd.Timestamp(year=year, month=1, day=1)))
            if start >= len(index):
                break
            lo = int(index.searchsorted(pd.Timestamp(year=year - self.train_years, month=1, day=1)))
            hi = start - self.h - 1                      # label of row hi is known at row hi + h < start
            rows = np.arange(lo + 2 * self.d, hi + 1, self.stride)
            net = self._fit(panels, label, ok, rows)
            if net is None:
                continue
            for r in rebal[(rebal >= start) & (index[rebal].year == year)]:
                cs = np.flatnonzero(ok[r])
                if not len(cs):
                    continue
                with torch.no_grad():
                    for i in range(0, len(cs), 512):
                        c = cs[i:i + 512]
                        x = torch.from_numpy(chart_batch(panels, np.full(len(c), r), c, self.d)).float().div_(255)
                        out[r, c] = torch.sigmoid(net(x).squeeze(-1)).numpy()
        return pd.DataFrame(out, index=index, columns=cols)


@dataclass(frozen=True)
class ChartCNN:
    """Portfolio strategy: hold the `top_n` stocks with the highest predicted probability
    of a positive `h`-day return (and short the `short_n` lowest), equal weight, at each
    month end. `panels['chart_cnn']` holds the scores from `ChartCNNModel.scores`."""
    top_n: int = 50
    short_n: int = 0
    every: str = "M"
    panels: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.top_n < 1 or self.short_n < 0:
            raise ValueError("require top_n >= 1, short_n >= 0")
        if self.every != "M":
            raise ValueError("scores are only computed at month ends (every='M')")

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        score = self.panels["chart_cnn"].reindex(index=closes.index, columns=closes.columns)
        w = _top_weights(score, closes, self.top_n, self.every, largest=True)
        if self.short_n:
            w = w - _top_weights(score, closes, self.short_n, self.every, largest=False)
        return w
