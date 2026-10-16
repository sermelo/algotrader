"""Deep learning statistical arbitrage (roadmap S8).

Guijarro-Ordonez, Pelger & Zanotti, "Deep learning statistical arbitrage" (Management
Science 2025; arXiv 2106.04028), benchmark model with PCA residuals:

1. **Residuals.** Each day t, on the stocks with complete returns over the previous
   `window` (252) days: K principal components of the correlation matrix give factor
   portfolios W = eigenvectors / volatility (Avellaneda & Lee 2010); factor returns
   F = W'R; loadings beta from OLS of each stock on F over the window; residual
   eps_t = R_t - beta F_t, i.e. eps = (I - beta W') R, a traded portfolio.
2. **Signal.** The last L = 30 cumulative residuals of each stock go through a 2-layer
   CNN (8 filters, width 2) and a transformer encoder (4 heads, feed-forward 2 x 8,
   dropout 0.25); a linear layer of the last step gives an allocation a_i.
3. **Weights.** Residual allocations map to stock weights w = (I - W beta') a,
   normalised to sum(|w|) = 1 (market neutral by construction, leverage 1).
4. **Training.** Maximise the Sharpe ratio of the daily returns net of costs
   (`cost_bps` x turnover + `short_bps` x short weight), Adam, learning rate 0.001,
   on the previous 1,000 days, refitted every 125 days, batches of 125 days.

Deviation: the paper uses CRSP's 550 largest stocks 1998-2016 with 5 bp + 1 bp a day on
shorts in training; we use point-in-time S&P 500 members, our cost model (about
2.5 bp per trade, 50 bp a year borrow) in training, and `epochs` is configurable.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- residuals

def pca_residuals(returns: pd.DataFrame, members: pd.DataFrame | None = None, k: int = 5,
                  window: int = 252):
    """Daily out-of-sample PCA residuals.

    Returns (eps, loadings): eps is a DataFrame like `returns` (NaN where a stock is not
    in that day's estimation set); loadings[t] = (tickers, W, beta) estimated on the
    `window` days before t, for mapping residual allocations at t to stock weights.
    """
    R = returns.to_numpy(dtype=float)
    T, N = R.shape
    eps = np.full((T, N), np.nan)
    loadings = {}
    mem = None
    if members is not None:
        mem = members.reindex(index=returns.index, columns=returns.columns).fillna(False).to_numpy(bool)
    for t in range(window, T):
        past = R[t - window:t]
        ok = np.isfinite(past).all(axis=0) & np.isfinite(R[t])
        if mem is not None:
            ok &= mem[t]
        idx = np.flatnonzero(ok)
        if len(idx) <= k + 1:
            continue
        X = past[:, idx]
        mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
        good = sd > 0
        idx, X, mu, sd = idx[good], X[:, good], mu[good], sd[good]
        Z = (X - mu) / sd
        corr = Z.T @ Z / (window - 1)
        vals, vecs = np.linalg.eigh(corr)
        V = vecs[:, ::-1][:, :k]                       # top k eigenvectors
        W = V / sd[:, None]                            # factor portfolio weights (N x k)
        F = X @ W                                      # factor returns over the window
        beta, *_ = np.linalg.lstsq(F, X, rcond=None)   # k x N
        beta = beta.T                                  # N x k
        r = R[t, idx]
        eps[t, idx] = r - beta @ (W.T @ r)
        loadings[returns.index[t]] = (returns.columns[idx], W, beta)
    return pd.DataFrame(eps, index=returns.index, columns=returns.columns), loadings


def stock_weights(a: np.ndarray, W: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Stock weights of the residual allocation a: w = (I - W beta') a, sum |w| = 1."""
    w = a - W @ (beta.T @ a)
    s = np.abs(w).sum()
    return w / s if s > 0 else w


# ---------------------------------------------------------------- model

def _torch():
    import torch
    return torch


def build_model(filters=8, heads=4, dropout=0.25, size=2, seed=0):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class CNNTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv1d(1, filters, size, padding=size - 1)
            self.c2 = nn.Conv1d(filters, filters, size, padding=size - 1)
            layer = nn.TransformerEncoderLayer(filters, heads, 2 * filters, dropout, batch_first=True)
            self.tr = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.out = nn.Linear(filters, 1)

        def forward(self, x):                          # x: (n, L)
            L = x.shape[1]
            h = torch.relu(self.c1(x[:, None, :]))[:, :, :L]
            h = torch.relu(self.c2(h))[:, :, :L]       # causal crop: (n, filters, L)
            h = self.tr(h.transpose(1, 2))             # (n, L, filters)
            return self.out(h[:, -1]).squeeze(-1)      # (n,)

    return CNNTransformer()


def _day_inputs(eps: np.ndarray, t: int, cols: np.ndarray, L: int):
    """Cumulative residuals of the L days ending at t (known at t's close) for `cols`."""
    win = eps[t - L + 1:t + 1, cols]
    ok = np.isfinite(win).all(axis=0)
    return np.cumsum(np.nan_to_num(win), axis=0).T, ok


@dataclass
class DeepArbModel:
    k: int = 5
    lookback: int = 30
    train_days: int = 1000
    refit_days: int = 125
    batch_days: int = 125
    epochs: int = 100
    lr: float = 1e-3
    cost_bps: float = 2.5
    short_bps_year: float = 50.0
    seed: int = 0
    log: list = field(default_factory=list)

    def _day_tensors(self, eps, rets, loadings, dates, pos, d):
        """Inputs, (W, beta), next-day returns and column ids for decision day d."""
        t = pos[d]
        cols_names, W, beta = loadings[d]
        cols = np.array([self._col[c] for c in cols_names])
        x, ok = _day_inputs(eps, t, cols, self.lookback)
        nxt = rets[t + 1, cols] if t + 1 < len(rets) else np.full(len(cols), np.nan)
        return x, ok, W, beta, nxt, cols

    def weights_panel(self, returns: pd.DataFrame, eps: pd.DataFrame, loadings: dict) -> pd.DataFrame:
        """Walk forward: refit every `refit_days` on the previous `train_days` decision days,
        then produce each day's stock weights (decided at the close, traded next open)."""
        torch = _torch()
        self._col = {c: i for i, c in enumerate(returns.columns)}
        E, R = eps.to_numpy(float), returns.to_numpy(float)
        dates = [d for d in returns.index if d in loadings]
        pos = {d: i for i, d in enumerate(returns.index)}
        dates = [d for d in dates if pos[d] >= self.lookback]
        out = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        model, start = None, self.train_days
        for i in range(start, len(dates), self.refit_days):
            model = self._fit([dates[j] for j in range(i - self.train_days, i)], E, R, loadings, pos)
            model.eval()
            with torch.no_grad():
                for d in dates[i:i + self.refit_days]:
                    x, ok, W, beta, _, cols = self._day_tensors(E, R, loadings, dates, pos, d)
                    a = model(torch.tensor(x, dtype=torch.float32)).numpy() * ok
                    out.loc[d, returns.columns[cols]] = stock_weights(a, W, beta)
        return out

    def _batches(self, days, E, R, loadings, pos):
        """Padded tensors per batch of `batch_days` days: inputs (B, N, L), mask (B, N),
        W and beta (B, N, k), next-day returns (B, N), global column ids (B, N)."""
        torch = _torch()
        days_data = []
        for d in days:
            x, ok, W, beta, nxt, cols = self._day_tensors(E, R, loadings, None, pos, d)
            if np.isfinite(nxt).any():
                days_data.append((x, ok, W, beta, np.nan_to_num(nxt), cols))
        dummy = len(self._col)                                  # padding column id
        out = []
        for b in range(0, len(days_data), self.batch_days):
            chunk = days_data[b:b + self.batch_days]
            B, N = len(chunk), max(len(c[5]) for c in chunk)
            X = np.zeros((B, N, self.lookback), np.float32)
            M = np.zeros((B, N), np.float32)
            Wt = np.zeros((B, N, self.k), np.float32)
            Bt = np.zeros((B, N, self.k), np.float32)
            Rt = np.zeros((B, N), np.float32)
            C = np.full((B, N), dummy, np.int64)
            for i, (x, ok, W, beta, nxt, cols) in enumerate(chunk):
                n = len(cols)
                X[i, :n], M[i, :n], Wt[i, :n], Bt[i, :n], Rt[i, :n], C[i, :n] = x, ok, W, beta, nxt, cols
            out.append(tuple(torch.from_numpy(a) for a in (X, M, Wt, Bt, Rt, C)))
        return out

    def _fit(self, days, E, R, loadings, pos):
        """Adam on the negative Sharpe of each batch's daily net returns (vectorised
        over the days of a batch; same objective as one day at a time)."""
        torch = _torch()
        model = build_model(seed=self.seed)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        batches = self._batches(days, E, R, loadings, pos)
        c, s = self.cost_bps / 1e4, self.short_bps_year / 1e4 / 252
        ncol = len(self._col) + 1
        for epoch in range(self.epochs):
            model.train()
            total = []
            for X, M, W, Bt, Rn, C in batches:
                B, N, L = X.shape
                a = model(X.reshape(B * N, L)).reshape(B, N) * M
                w = a - torch.einsum("bnk,bk->bn", W, torch.einsum("bnk,bn->bk", Bt, a))
                w = w * (C < ncol - 1)                          # padding rows
                w = w / (w.abs().sum(dim=1, keepdim=True) + 1e-12)
                full = torch.zeros(B, ncol).scatter(1, C, w)[:, :-1]
                cost = s * torch.clamp(full, max=0).abs().sum(dim=1)
                turn = torch.cat([torch.zeros(1), (full[1:] - full[:-1]).abs().sum(dim=1)])
                r = (w * Rn).sum(dim=1) - cost - c * turn
                loss = -r.mean() / (r.std() + 1e-8)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total.append(float(-loss.detach()))
            self.log.append((days[-1], epoch, float(np.mean(total))))
        return model


@dataclass(frozen=True)
class DeepArb:
    """Portfolio strategy holding the precomputed deep-arbitrage weights (panel in
    `panels['deep_arb']`), rebalanced daily."""
    k: int = 5
    panels: dict = field(default=None, repr=False, compare=False)

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        w = self.panels[f"deep_arb_k{self.k}"]
        return w.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
