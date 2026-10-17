"""Walk-forward gradient boosting on the minute-feature dataset (roadmap S14).

The question: can a model pick, among the most volatile stocks at a decision minute, the one (or three) whose return
over the next `h` minutes beats the cost of the trade? Design, fixed before looking at any test year:

* Universe: at every decision (day and minute), the stocks in the top `1 - vol_rank` of the cross-section by
  30-minute volatility (the tenth where typical moves are several times the cost).
* Target: the forward return minus the average of all stocks at that decision (market drift removed), clipped to
  +-`clip`. One model per horizon.
* Features: the 23 minute features of `minutefeat` plus the volatility-scaled moves of `minutescan.derived`.
* Model: `HistGradientBoostingRegressor` with strong regularisation and fixed settings (the signal is tiny and
  the data noisy); no tuning on the test years.
* Walk-forward: for each test year Y, fit on all earlier years only (an expanding window) and predict year Y.
  Nothing from year Y, or later, reaches the fit.

Predictions are scored like any other pick (`minutescan.RuleScan` on the `pred_*` columns): net of the round-trip cost,
by decision minute and number of picks.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .minutefeat import FEATURES, LABELS
from .minutescan import DERIVED, derived

MODEL_FEATURES = FEATURES + DERIVED
HORIZONS = ("fwd30", "fwd90", "fwd240", "fwd_close")
PARAMS = dict(learning_rate=0.05, max_iter=150, max_bins=64, max_leaf_nodes=15, min_samples_leaf=1000, l2_regularization=10.0, random_state=0)


def prepare(df: pd.DataFrame, vol_rank: float = 0.9, horizons=HORIZONS, derive=derived) -> pd.DataFrame:
    """The universe rows with `x_<label>` (return minus the decision's cross-sectional mean) targets; the mean is taken over
    all stocks of the decision before the volatility filter."""
    df = derive(df) if derive else df
    g = df.groupby(["date", "minute"], sort=False)
    for lab in horizons:
        df[f"x_{lab}"] = df[lab] - g[lab].transform("mean")
    rank = g["vol30"].rank(pct=True)
    return df[rank >= vol_rank]


@dataclass
class WalkForward:
    first_test_year: int = 2013
    horizons: tuple = HORIZONS
    features: list = field(default_factory=lambda: list(MODEL_FEATURES))
    params: dict = field(default_factory=lambda: dict(PARAMS))
    clip: float = 0.03
    min_train_rows: int = 5000
    shuffle_seed: int = None                 # placebo: permute the training targets, which must leave nothing to learn
    log: list = field(default_factory=list)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """Predictions `pred_<label>` for the rows of `test`, from models fitted on `train` only."""
        out = test[["date", "minute", "ticker"] + list(LABELS)].copy()
        X, Xt = train[self.features].to_numpy(np.float32), test[self.features].to_numpy(np.float32)
        for lab in self.horizons:
            y = train[f"x_{lab}"].clip(-self.clip, self.clip).to_numpy(np.float32)
            ok = np.isfinite(y)
            if self.shuffle_seed is not None:
                y = np.random.default_rng(self.shuffle_seed).permutation(y)
            if ok.sum() < self.min_train_rows:
                out[f"pred_{lab}"] = np.nan
                continue
            model = HistGradientBoostingRegressor(**self.params).fit(X[ok], y[ok])
            out[f"pred_{lab}"] = model.predict(Xt)
            self.log.append((int(test["date"].dt.year.iloc[0]), lab, int(ok.sum())))
        return out

    def run(self, year_frames: dict) -> pd.DataFrame:
        """`year_frames`: calendar year -> `prepare`d frame. Test years from `first_test_year`, each fitted on the earlier years."""
        years = sorted(year_frames)
        parts = []
        for y in years:
            if y < self.first_test_year:
                continue
            train = pd.concat([year_frames[k] for k in years if k < y], ignore_index=True)
            if train.empty or year_frames[y].empty:
                continue
            parts.append(self.fit_predict(train, year_frames[y]))
        return pd.concat(parts, ignore_index=True)


def information_coefficient(preds: pd.DataFrame, label: str) -> pd.Series:
    """Rank correlation between the prediction and the realised return across stocks at each decision, averaged by year."""
    g = preds.groupby(["date", "minute"], sort=False)
    z = lambda s: (g[s].rank(pct=True) - 0.5) * np.sqrt(12)
    prod = (z(f"pred_{label}") * z(label)).groupby([preds["date"], preds["minute"]]).mean()
    return prod.groupby(prod.index.get_level_values("date").year).mean()


def threshold_table(preds: pd.DataFrame, label: str, k: int, cost: float, taus=(0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003)) -> pd.DataFrame:
    """One trade per decision: the `k` stocks with the highest prediction, taken only when their mean prediction is at least `tau`.
    Per threshold: trades a year, mean gross and net (after `cost`) return of those trades in basis points."""
    key = ["date", "minute"]
    g = preds.groupby(key, sort=False)
    rank = g[f"pred_{label}"].rank(method="first", ascending=False)
    top = preds[rank <= k].groupby(key)[[f"pred_{label}", label]].mean().dropna()
    days = preds["date"].nunique() or 1
    rows = []
    for tau in taus:
        sel = top[top[f"pred_{label}"] >= tau]
        rows.append({"tau_bp": tau * 1e4, "trades": len(sel), "trades_per_year": len(sel) / (days / 252), "gross_bp": sel[label].mean() * 1e4 if len(sel) else np.nan,
                     "net_bp": (sel[label].mean() - cost) * 1e4 if len(sel) else np.nan,
                     "net_t": (sel[label] - cost).mean() / ((sel[label] - cost).std() / np.sqrt(len(sel))) if len(sel) > 2 else np.nan})
    return pd.DataFrame(rows)


def best_of_day(preds: pd.DataFrame, label: str, cost: float, taus=(0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005)) -> pd.DataFrame:
    """One trade a day, the one the model is most sure of: across all of a day's decisions, the stock with the highest
    prediction; taken only if that prediction is at least `tau`. Per threshold: trades a year, the mean gross and net
    (after `cost`) return in basis points, the t-statistic of the net return and the share of winning trades."""
    p = preds[["date", "minute", "ticker", f"pred_{label}", label]].dropna()
    best = p.loc[p.groupby("date")[f"pred_{label}"].idxmax()]
    days = preds["date"].nunique() or 1
    rows = []
    for tau in taus:
        sel = best[best[f"pred_{label}"] >= tau]
        net = sel[label] - cost
        rows.append({"tau_bp": tau * 1e4, "trades": len(sel), "trades_per_year": len(sel) / (days / 252),
                     "gross_bp": sel[label].mean() * 1e4 if len(sel) else np.nan, "net_bp": net.mean() * 1e4 if len(sel) else np.nan,
                     "net_t": net.mean() / (net.std() / np.sqrt(len(sel))) if len(sel) > 2 and net.std() > 0 else np.nan,
                     "win": float((net > 0).mean()) if len(sel) else np.nan})
    return pd.DataFrame(rows)


def replay_fills(bars: pd.DataFrame, d: int, hold: int) -> dict:
    """Re-price one trade on the raw minute bars (index = minute after 09:30; columns open, high, low, close; only minutes with
    trades) under several fills; returns the gross return of each, or None if the entry minute has no trade.

    `base`: buy the open of minute d, sell the open of the first bar at or after d + hold (else the last close) - what the labels
    assume. `worst_bar`: buy at that bar's high and sell at the exit bar's low. `delay1`, `delay2`, `delay5`: buy at the open of the
    first bar at or after d + 1, 2 or 5, sell as `base`. `range_slip`: buy a quarter of the entry bar's range above its open and
    sell a quarter of the exit bar's range below its open."""
    if d not in bars.index:
        return None
    entry = bars.loc[d]
    later = bars[bars.index >= d + hold]
    if len(later):
        ex = later.iloc[0]
        exit_open, exit_low, exit_rng = ex["open"], ex["low"], ex["high"] - ex["low"]
    else:
        ex = bars.iloc[-1]
        exit_open = exit_low = ex["close"]
        exit_rng = 0.0
    out = {"base": exit_open / entry["open"] - 1,
           "worst_bar": exit_low / entry["high"] - 1,
           "range_slip": (exit_open - 0.25 * exit_rng) / (entry["open"] + 0.25 * (entry["high"] - entry["low"])) - 1}
    for k in (1, 2, 5):
        e = bars[bars.index >= d + k]
        out[f"delay{k}"] = exit_open / e.iloc[0]["open"] - 1 if len(e) else np.nan
    return out
