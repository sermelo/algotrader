"""Where could an intraday edge be? Two scans of the minute-feature dataset (`minutefeat.py`).

`BinTable`: how big are the moves, by a condition (stock volatility, volume surge, gap, time of day ...)? For each
bin of the condition and each holding horizon: windows, mean absolute move, mean signed return and the share of
windows that rise or fall by more than each cost threshold. Conditions that concentrate large moves are where
the typical move is several times the cost.

`SignalScan`: does any feature rank stocks by their later return? On each decision (one day and minute) the
stocks are ranked on a feature and on a forward return; the rank correlation (information coefficient, IC) is
averaged by day, with its t-statistic and how many calendar years it is positive; plus the cross-sectional excess
return (over the average stock of that decision) of the top and bottom tenth and of the single top and bottom
stock, in basis points. Excess returns remove the market's own drift, so they measure stock picking.
"""
import numpy as np
import pandas as pd

from .minutefeat import LABELS

SCAN_FEATURES = ["ret5", "ret15", "ret30", "ret60", "ret_open", "gap", "prev_ret", "vol30", "dvol20", "rvol_day",
                 "rvol30", "vwap_dist", "range_pos", "range_bp", "active30", "rel30", "rel_open", "price", "logdv20"]
CONDITIONS = {                                    # name -> function of the dataset giving the conditioning value
    "dvol20": lambda d: d["dvol20"], "vol30": lambda d: d["vol30"], "range_bp": lambda d: d["range_bp"],
    "rvol30": lambda d: d["rvol30"], "rvol_day": lambda d: d["rvol_day"], "abs_gap": lambda d: d["gap"].abs(),
    "abs_ret_open": lambda d: d["ret_open"].abs(), "abs_rel30": lambda d: d["rel30"].abs(),
    "price": lambda d: d["price"], "logdv20": lambda d: d["logdv20"], "minute": lambda d: d["minute"],
}
DISCRETE = {"minute"}
THRESHOLDS = (0.0008, 0.0014, 0.0019)


def quantile_edges(values: np.ndarray, bins: int = 10) -> np.ndarray:
    """Inner bin edges (bins - 1 of them) at equal-count quantiles."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))


class BinTable:
    """Move statistics by bin of one condition (equal-count bins from `edges`, or each value of a discrete one)."""

    def __init__(self, name: str, edges: np.ndarray = None, labels=LABELS, thresholds=THRESHOLDS):
        self.name, self.edges, self.labels, self.thr = name, edges, list(labels), tuple(thresholds)
        self.discrete = name in DISCRETE
        self.keys = {}                                              # discrete value -> bin index
        k = (len(edges) + 1) if edges is not None else 0
        self._alloc(k)

    def _alloc(self, k):
        shape = (k, len(self.labels))
        self.n, self.abs, self.sum = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.up, self.either = np.zeros(shape + (len(self.thr),)), np.zeros(shape + (len(self.thr),))
        self.vsum = np.zeros(k)

    def _grow(self, k):
        if k > len(self.n):
            old = (self.n, self.abs, self.sum, self.up, self.either, self.vsum)
            self._alloc(k)
            for new, o in zip((self.n, self.abs, self.sum, self.up, self.either, self.vsum), old):
                new[:len(o)] = o

    def add(self, df: pd.DataFrame):
        v = CONDITIONS[self.name](df).to_numpy(dtype=float)
        ok = np.isfinite(v)
        if self.discrete:
            for u in np.unique(v[ok]):
                self.keys.setdefault(float(u), len(self.keys))
            idx = np.array([self.keys.get(float(x), -1) for x in v])
            self._grow(len(self.keys))
        else:
            idx = np.digitize(v, self.edges)
        for j, lab in enumerate(self.labels):
            y = df[lab].to_numpy(dtype=float)
            m = ok & np.isfinite(y) & (idx >= 0)
            i, yy = idx[m], y[m]
            k = len(self.n)
            self.n[:, j] += np.bincount(i, minlength=k)
            self.abs[:, j] += np.bincount(i, weights=np.abs(yy), minlength=k)
            self.sum[:, j] += np.bincount(i, weights=yy, minlength=k)
            for t, c in enumerate(self.thr):
                self.up[:, j, t] += np.bincount(i, weights=(yy > c), minlength=k)
                self.either[:, j, t] += np.bincount(i, weights=(np.abs(yy) > c), minlength=k)
        mv = ok & (idx >= 0)
        self.vsum += np.bincount(idx[mv], weights=v[mv], minlength=len(self.n))
        self.vcount = getattr(self, "vcount", np.zeros(len(self.n)))
        if len(self.vcount) < len(self.n):
            self.vcount = np.pad(self.vcount, (0, len(self.n) - len(self.vcount)))
        self.vcount += np.bincount(idx[mv], minlength=len(self.n))

    def table(self) -> pd.DataFrame:
        rows = []
        order = np.argsort(list(self.keys)) if self.discrete else range(len(self.n))
        values = sorted(self.keys) if self.discrete else [None] * len(self.n)
        for pos, b in enumerate(order):
            for j, lab in enumerate(self.labels):
                n = self.n[b, j]
                if not n:
                    continue
                r = {"condition": self.name, "bin": pos, "bin_value": values[pos] if self.discrete else self.vsum[b] / max(self.vcount[b], 1),
                     "label": lab, "n": int(n), "mean_abs": self.abs[b, j] / n, "mean": self.sum[b, j] / n}
                for t, c in enumerate(self.thr):
                    r[f"up_{c}"] = self.up[b, j, t] / n
                    r[f"either_{c}"] = self.either[b, j, t] / n
                rows.append(r)
        return pd.DataFrame(rows)


class SignalScan:
    """Rank correlation and top/bottom picks of each feature against each forward return (see module doc)."""

    def __init__(self, features=SCAN_FEATURES, labels=LABELS):
        self.features, self.labels = list(features), list(labels)
        self.dates, self.ic, self.top10, self.bot10, self.top1, self.bot1, self.top1raw = [], [], [], [], [], [], []

    @staticmethod
    def _bounds(key):
        return np.flatnonzero(np.r_[True, key[1:] != key[:-1]])

    def add(self, df: pd.DataFrame):
        """`df` sorted by date and minute (as `minutefeat.process` writes it)."""
        g = df.groupby(["date", "minute"], sort=False)
        gid = g.ngroup().to_numpy()
        starts = self._bounds(gid)
        sums = lambda a: np.add.reduceat(a, starts, axis=0)
        F, L = len(self.features), len(self.labels)
        rk = np.column_stack([g[f].rank(pct=True).to_numpy() for f in self.features])         # (rows, F), NaN where missing
        yv = df[self.labels].to_numpy(dtype=float)
        ry = np.column_stack([g[l].rank(pct=True).to_numpy() for l in self.labels])
        zf, zy = (rk - 0.5) * np.sqrt(12), (ry - 0.5) * np.sqrt(12)
        fin_f = np.isfinite(zf)
        zf0 = np.where(fin_f, zf, 0.0)
        n_g = len(starts)
        ic = np.zeros((n_g, F, L)); t10 = np.zeros_like(ic); b10 = np.zeros_like(ic); t1 = np.zeros_like(ic); b1 = np.zeros_like(ic); t1r = np.zeros_like(ic)
        sizes = np.diff(np.r_[starts, len(df)])
        gmin = np.repeat(np.minimum.reduceat(np.where(fin_f, rk, np.inf), starts, axis=0), sizes, axis=0)
        r0 = np.nan_to_num(rk, nan=-1.0)
        masks = {"t10": (r0 > 0.9) & fin_f, "b10": (r0 <= 0.1) & fin_f, "t1": (r0 >= 1 - 1e-9) & fin_f, "b1": (rk <= gmin + 1e-12) & fin_f}
        for j in range(L):
            fy = np.isfinite(yv[:, j])
            y0 = np.where(fy, yv[:, j], 0.0)
            zy0 = np.where(fy, zy[:, j], 0.0)
            gm = sums(y0) / np.maximum(sums(fy.astype(float)), 1)                            # group mean return
            both = fin_f & fy[:, None]
            nb = np.maximum(sums(both.astype(float)), 1)
            ic[:, :, j] = sums(zf0 * zy0[:, None] * both) / nb
            for key, store in (("t10", t10), ("b10", b10), ("t1", t1), ("b1", b1)):
                msk = masks[key] & fy[:, None]
                have = sums(msk.astype(float))
                mean_y = sums(msk * y0[:, None]) / np.maximum(have, 1)
                store[:, :, j] = np.where(have > 0, mean_y - gm[:, None], np.nan)
                if key == "t1":
                    t1r[:, :, j] = np.where(have > 0, mean_y, np.nan)
        self.dates.append(df["date"].to_numpy()[starts])
        for store, arr in ((self.ic, ic), (self.top10, t10), (self.bot10, b10), (self.top1, t1), (self.bot1, b1), (self.top1raw, t1r)):
            store.append(arr)

    def result(self) -> pd.DataFrame:
        dates = pd.DatetimeIndex(np.concatenate(self.dates))
        cat = lambda s: np.concatenate(s, axis=0)
        arrs = {k: cat(v) for k, v in (("ic", self.ic), ("top10", self.top10), ("bot10", self.bot10), ("top1", self.top1),
                                       ("bot1", self.bot1), ("top1raw", self.top1raw))}
        days, years = dates.normalize(), dates.year
        out = []
        for fi, f in enumerate(self.features):
            for li, l in enumerate(self.labels):
                r = {"feature": f, "label": l}
                daily = pd.Series(arrs["ic"][:, fi, li], index=days).groupby(level=0).mean().dropna()
                r["days"] = len(daily)
                r["ic"] = daily.mean()
                r["ic_t"] = r["ic"] / (daily.std() / np.sqrt(len(daily))) if len(daily) > 2 and daily.std() > 0 else np.nan
                yearly = pd.Series(arrs["ic"][:, fi, li], index=years).groupby(level=0).mean()
                r["years_positive"] = int((yearly > 0).sum())
                r["years"] = int(yearly.notna().sum())
                for k in ("top10", "bot10", "top1", "bot1", "top1raw"):
                    d = pd.Series(arrs[k][:, fi, li], index=days).groupby(level=0).mean().dropna()
                    r[k + "_bp"] = d.mean() * 1e4 if len(d) else np.nan
                    r[k + "_t"] = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
                out.append(r)
        return pd.DataFrame(out)


class RuleScan:
    """Simple stock-picking rules, one trade per decision: among the stocks passing a filter, hold the `k` with the
    lowest ("low") or highest ("high") value of a score, for each forward return. Per decision (day and minute) it
    keeps the picks' mean return and the filtered set's mean return; `result` gives, by rule, decision minute and
    horizon: the picks' mean, the excess over the filtered set, the same net of the round-trip cost of `k` slots
    (`costs[k]`), their t-statistics over days and the share of winning days. The number of rules tested is what a
    t-statistic has to be judged against."""

    def __init__(self, scores, filters: dict, ks=(1, 3), labels=LABELS, costs=None):
        self.scores, self.filters, self.ks, self.labels = list(scores), filters, tuple(ks), list(labels)
        self.costs = costs or {1: 0.0008, 3: 0.0014}
        self.picks, self.base = {}, {}

    def add(self, df: pd.DataFrame):
        key = ["date", "minute"]
        for fname, fn in self.filters.items():
            sub = df[fn(df)]
            if sub.empty:
                continue
            g = sub.groupby(key, sort=False)
            self.base.setdefault(fname, []).append(g[self.labels].mean())
            for s in self.scores:
                for direction in ("low", "high"):
                    r = g[s].rank(method="first", ascending=direction == "low")
                    for k in self.ks:
                        pick = sub[r <= k].groupby(key, sort=False)[self.labels].mean()
                        self.picks.setdefault((fname, s, direction, k), []).append(pick)

    def result(self) -> pd.DataFrame:
        rows = []
        base = {f: pd.concat(v) for f, v in self.base.items()}
        for (fname, s, direction, k), parts in self.picks.items():
            pick = pd.concat(parts)
            b = base[fname].reindex(pick.index)
            for lab in self.labels:
                for minute, idx in pick.groupby(level="minute").groups.items():
                    x = pick.loc[idx, lab].dropna()
                    if len(x) < 30:
                        continue
                    xb = b.loc[x.index, lab]
                    ex, net = x - xb, x - self.costs[k]
                    years = x.index.get_level_values("date").year
                    t = lambda v: v.mean() / (v.std() / np.sqrt(len(v))) if v.std() > 0 else np.nan
                    rows.append({"filter": fname, "score": s, "pick": direction, "k": k, "minute": int(minute), "label": lab, "days": len(x),
                                 "gross_bp": x.mean() * 1e4, "filter_mean_bp": xb.mean() * 1e4, "excess_bp": ex.mean() * 1e4, "excess_t": t(ex),
                                 "net_bp": net.mean() * 1e4, "net_t": t(net), "win_net": float((net > 0).mean()),
                                 "net_years_positive": int((net.groupby(years).mean() > 0).sum()), "years": int(pd.Series(years).nunique())})
        return pd.DataFrame(rows)
