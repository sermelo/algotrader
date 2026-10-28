"""Point-in-time fundamental signals from SEC XBRL facts (roadmap S3).

Every value is used only from the trading day after it was filed (`filed` + 1 day),
never from its period end, so no signal uses information before it was public. Companies
switch XBRL tags over the years (e.g. to the equity tag that includes non-controlling
interests), so the preferred tag is chosen for each reported period, not once per company.
A ratio's two parts come from the same period end, and a value counts only until
`MAX_AGE` (18 months) after its period end, the Fama-French convention: a company that stops
reporting a fact drops out instead of keeping a stale value.

Signals (daily panels, dates x tickers):
- **book_to_market** (Fama & French 1992): latest reported stockholders' equity (10-K or
  10-Q) / market capitalisation, market cap = latest reported shares outstanding x
  unadjusted close. Negative book equity -> NaN.
- **gross_profitability** (Novy-Marx 2013): gross profit (reported, else revenue - cost
  of revenue) / total assets, from the latest annual report (10-K, fiscal-year values).
  Companies reporting neither (e.g. most banks) have no value.
- **asset_growth** (Cooper, Gulen & Schill 2008): total assets at the latest annual
  report / at the fiscal year end 330-400 days before it - 1 (low growth is the long side).
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
MAX_AGE = pd.Timedelta(days=548)          # 18 months after the period end


def _known(facts: pd.DataFrame, tags, annual=False, duration_days=None) -> pd.DataFrame:
    """Rows (cik, available, end, value), one per company, period end and filing: where a filing
    reports the period under several of `tags`, the first of them. Amendments and later comparatives
    are their own rows, usable from their own filing date."""
    f = facts[facts["tag"].isin(tags)]
    if annual:
        f = f[f["form"].isin(ANNUAL_FORMS)]
    if duration_days is not None:
        d = (f["end"] - f["start"]).dt.days
        f = f[(d >= duration_days[0]) & (d <= duration_days[1])]
    rank = {t: i for i, t in enumerate(tags)}
    f = f.assign(_r=f["tag"].map(rank)).sort_values("_r", kind="stable")
    f = f.drop_duplicates(["cik", "end", "filed"]).sort_values("filed", kind="stable")
    f = f.assign(available=f["filed"] + pd.Timedelta(days=1))
    return f[["cik", "available", "end", "value"]].reset_index(drop=True)


def _combine(a: pd.DataFrame, b: pd.DataFrame, fn) -> pd.DataFrame:
    """Rows of `fn(a value, b value)` per company and period end (NaN where a side has none yet):
    one row whenever either side gets a new value for that period, from each side's latest value
    filed by then. Non-finite results are dropped."""
    keys = (pd.concat([a, b])[["cik", "end", "available"]].drop_duplicates()
            .sort_values("available", kind="stable").reset_index(drop=True))

    def asof(rows):
        m = pd.merge_asof(keys, rows.sort_values("available", kind="stable"), on="available", by=["cik", "end"])
        return m["value"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = keys.assign(value=fn(asof(a), asof(b)))
    return out[np.isfinite(out["value"])].reset_index(drop=True)


def _previous_year(rows: pd.DataFrame) -> pd.DataFrame:
    """`rows` relabelled to the company's next fiscal year end (330-400 days later), where it reports one."""
    ends = rows[["cik", "end"]].drop_duplicates()
    pairs = ends.merge(ends, on="cik", suffixes=("", "_next"))
    gap = (pairs["end_next"] - pairs["end"]).dt.days
    pairs = pairs[(gap >= 330) & (gap <= 400)].sort_values("end").drop_duplicates(["cik", "end_next"], keep="last")
    moved = rows.merge(pairs, on=["cik", "end"])
    return moved.assign(end=moved["end_next"])[["cik", "available", "end", "value"]]


def _latest_panel(rows: pd.DataFrame, cik_of: dict, index: pd.DatetimeIndex, max_age=MAX_AGE) -> pd.DataFrame:
    """Daily panel: for each ticker, the value with the latest period end among the
    rows available by that day, while that period ended at most `max_age` before."""
    days = index.to_series()
    out = {}
    for ticker, cik in cik_of.items():
        r = rows[rows["cik"] == cik].sort_values("available", kind="stable")
        if r.empty:
            continue
        vals, best_end, best_val = [], pd.Timestamp.min, np.nan
        for _, x in r.iterrows():
            if x["end"] >= best_end:
                best_end, best_val = x["end"], x["value"]
            vals.append((x["available"], best_val, best_end))
        s = pd.DataFrame(vals, columns=["available", "value", "end"]).drop_duplicates("available", keep="last")
        s = s.set_index("available").sort_index()
        s = s.reindex(index.union(s.index)).ffill().reindex(index)
        out[ticker] = s["value"].where(days - s["end"] <= max_age)
    return pd.DataFrame(out, index=index)


def signals(facts: pd.DataFrame, cik_map: pd.DataFrame, raw_close: pd.DataFrame) -> dict:
    """The four signal panels on `raw_close`'s dates and tickers."""
    index = raw_close.index
    cik_of = {t: int(c) for t, c in cik_map["cik"].dropna().items() if t in raw_close.columns}
    panel = lambda rows: _latest_panel(rows, cik_of, index)
    equity = panel(_known(facts, EQUITY))
    shares = panel(_known(facts, SHARES))
    mcap = shares * raw_close.reindex(columns=equity.columns.union(shares.columns))
    btm = (equity / mcap.reindex(columns=equity.columns)).where(lambda x: x > 0)
    year = (330, 400)
    annual = lambda tags, days=None: _known(facts, tags, annual=True, duration_days=days)
    assets = annual(("Assets",))
    rev_cost = _combine(annual(REVENUE, year), annual(COST, year), np.subtract)
    gross = _combine(annual(("GrossProfit",), year), rev_cost, lambda g, rc: np.where(np.isnan(g), rc, g))
    gp = panel(_combine(gross, assets, np.divide))
    growth = panel(_combine(assets, _previous_year(assets), lambda a, p: a / p - 1))
    op = annual(("OperatingIncomeLoss",), year)
    opbe = panel(_combine(op, annual(EQUITY), lambda o, e: np.where(e > 0, o / e, np.nan)))
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
