"""Point-in-time fundamental signals from SEC XBRL facts (roadmap S3).

Every value is used only from the trading day after it was filed (`filed` + 1 day),
never from its period end, so no signal uses information before it was public.

Signals (daily panels, dates x tickers):
- **book_to_market** (Fama & French 1992): latest reported stockholders' equity (10-K or
  10-Q) / market capitalisation, market cap = latest reported shares outstanding x
  unadjusted close. Negative book equity -> NaN.
- **gross_profitability** (Novy-Marx 2013): gross profit (reported, else revenue - cost
  of revenue) / total assets, from the latest annual report (10-K, fiscal-year values).
  Companies reporting neither (e.g. most banks) have no value.
- **asset_growth** (Cooper, Gulen & Schill 2008): total assets at the latest annual
  report / at the one before - 1 (low growth is the long side).
- **operating_profitability** (Fama & French 2015, as in their RMW factor): latest annual
  operating income / latest annual stockholders' equity (positive equity only); covers
  banks and insurers too.

Strategy: `FundamentalSort` holds the `top_n` tickers with the best signal (and shorts
the `short_n` worst), rebalanced at period ends, like the other cross-sectional ones.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import _top_weights

EQUITY = ("StockholdersEquity",
          "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
SHARES = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")
REVENUE = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
COST = ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "20-F", "40-F")


def _known(facts: pd.DataFrame, tags, annual=False, duration_days=None) -> pd.DataFrame:
    """Rows (cik, available, end, value) for the first tag present per company, one per
    (cik, end) keeping the first filing (later amendments would be look-ahead for the
    earlier date and are added as their own rows by their own filing date)."""
    f = facts[facts["tag"].isin(tags)]
    if annual:
        f = f[f["form"].isin(ANNUAL_FORMS)]
    if duration_days is not None:
        d = (f["end"] - f["start"]).dt.days
        f = f[(d >= duration_days[0]) & (d <= duration_days[1])]
    # prefer the first tag in `tags` that a company uses
    rank = {t: i for i, t in enumerate(tags)}
    f = f.assign(_r=f["tag"].map(rank)).sort_values(["cik", "_r"])
    best = f.groupby("cik")["_r"].transform("min")
    f = f[f["_r"] == best]
    f = f.sort_values("filed").drop_duplicates(["cik", "end", "filed"])
    f = f.assign(available=f["filed"] + pd.Timedelta(days=1))
    return f[["cik", "available", "end", "value"]]


def _latest_panel(rows: pd.DataFrame, cik_of: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily panel: for each ticker, the value with the latest period end among the
    rows available by that day."""
    out = {}
    for ticker, cik in cik_of.items():
        r = rows[rows["cik"] == cik].sort_values("available")
        if r.empty:
            continue
        vals, best_end, best_val = [], pd.Timestamp.min, np.nan
        for _, x in r.iterrows():
            if x["end"] >= best_end:
                best_end, best_val = x["end"], x["value"]
            vals.append((x["available"], best_val))
        s = pd.Series(dict(vals))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[ticker] = s.reindex(index.union(s.index)).ffill().reindex(index)
    return pd.DataFrame(out, index=index)


def _annual_pair_panel(rows: pd.DataFrame, cik_of: dict, index) -> tuple:
    """(latest, previous) annual values by availability date, per ticker."""
    latest, prev = {}, {}
    for ticker, cik in cik_of.items():
        r = rows[rows["cik"] == cik].sort_values("available")
        if r.empty:
            continue
        seen, a, b = {}, [], []
        for _, x in r.iterrows():
            seen[x["end"]] = x["value"] if x["end"] not in seen else seen[x["end"]]
            ends = sorted(seen)
            a.append((x["available"], seen[ends[-1]]))
            b.append((x["available"], seen[ends[-2]] if len(ends) > 1 else np.nan))
        for store, vals in ((latest, a), (prev, b)):
            s = pd.Series(dict(vals))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            store[ticker] = s.reindex(index.union(s.index)).ffill().reindex(index)
    return pd.DataFrame(latest, index=index), pd.DataFrame(prev, index=index)


def signals(facts: pd.DataFrame, cik_map: pd.DataFrame, raw_close: pd.DataFrame) -> dict:
    """The three signal panels on `raw_close`'s dates and tickers."""
    index = raw_close.index
    cik_of = {t: int(c) for t, c in cik_map["cik"].dropna().items() if t in raw_close.columns}
    equity = _latest_panel(_known(facts, EQUITY), cik_of, index)
    shares = _latest_panel(_known(facts, SHARES), cik_of, index)
    mcap = shares * raw_close.reindex(columns=equity.columns.union(shares.columns))
    btm = (equity / mcap.reindex(columns=equity.columns)).where(lambda x: x > 0)
    year = (330, 400)
    rev = _latest_panel(_known(facts, REVENUE, annual=True, duration_days=year), cik_of, index)
    cost = _latest_panel(_known(facts, COST, annual=True, duration_days=year), cik_of, index)
    assets_rows = _known(facts, ("Assets",), annual=True)
    assets, assets_prev = _annual_pair_panel(assets_rows, cik_of, index)
    gross = _latest_panel(_known(facts, ("GrossProfit",), annual=True, duration_days=year),
                          cik_of, index)
    cols = rev.columns.union(gross.columns)
    gp_value = gross.reindex(columns=cols).combine_first(
        rev.reindex(columns=cols) - cost.reindex(columns=cols))
    gp = gp_value / assets.reindex(columns=cols)
    growth = assets / assets_prev - 1
    op = _latest_panel(_known(facts, ("OperatingIncomeLoss",), annual=True, duration_days=year),
                       cik_of, index)
    eq_annual = _latest_panel(_known(facts, EQUITY, annual=True), cik_of, index)
    eq_annual = eq_annual.reindex(columns=op.columns)
    opbe = (op / eq_annual).where(eq_annual > 0)
    clean = lambda df: df.reindex(columns=raw_close.columns).replace([np.inf, -np.inf], np.nan)
    return {"book_to_market": clean(btm), "gross_profitability": clean(gp),
            "asset_growth": clean(growth), "operating_profitability": clean(opbe)}


@dataclass(frozen=True)
class FundamentalSort:
    """Hold the `top_n` tickers with the highest `signal` (lowest if not `largest`),
    and short the `short_n` at the other end; equal weight, rebalanced at period ends.
    `panels` holds the precomputed signal panels (see `signals`)."""
    signal: str = "book_to_market"
    top_n: int = 50
    short_n: int = 0
    largest: bool = True
    every: str = "M"
    panels: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.top_n < 1 or self.short_n < 0:
            raise ValueError("require top_n >= 1, short_n >= 0")
        if self.every not in ("D", "W", "M"):
            raise ValueError("every must be 'D', 'W' or 'M'")

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        score = self.panels[self.signal].reindex(index=closes.index, columns=closes.columns)
        w = _top_weights(score, closes, self.top_n, self.every, largest=self.largest)
        if self.short_n:
            w = w - _top_weights(score, closes, self.short_n, self.every, largest=not self.largest)
        return w
