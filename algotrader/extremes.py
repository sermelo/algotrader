"""Biggest monthly winners and losers among S&P 500 members (TASKS 11, step 1: descriptive, no trading).

For every month, the members' returns from one month end to the next; the top and bottom 10 and deciles;
and what was known at the first month end: past returns, volatility, the largest daily return (Bali,
Cakici & Whitelaw 2011), distance from the 52-week high, dollar volume (size), SEC fundamentals, whether
an earnings release is due (one three months earlier: the quarterly cycle), recent index entry and
industry. For each feature, the lift: how much more often a stock in a bucket of it (quintile, or
category) lands in a tail than the average member, pooled and by year. Extreme months come mostly from
volatility and news, which fill both tails; what matters is an asymmetric lift (more often in one tail
than in the other), stable across years. `scripts/monthly_extremes.py` runs it on the long history.
"""
import numpy as np
import pandas as pd

GROUPS = ("top10", "bottom10", "top_decile", "bottom_decile")
QUINTILES = 5


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def monthly_returns(closes: pd.DataFrame, ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Return from each month end to the next (rows: the starting month end). A stock needs a close on
    the starting day; one whose series ends during the month counts to its last close (a delisting).
    A non-positive price at either end (a broken adjustment) gives no return."""
    start = closes.reindex(ends[:-1])
    end = closes.ffill().reindex(ends[1:])
    end.index = ends[:-1]
    return (end / start - 1).where((start > 0) & (end > 0))


def price_features(closes: pd.DataFrame, dollar_volume: pd.DataFrame, ends: pd.DatetimeIndex) -> dict:
    """Price features at each month end, from data up to that day's close (dates x tickers each)."""
    ret = closes.pct_change(fill_method=None)
    feats = {
        "ret_1m": closes / closes.shift(21) - 1,
        "ret_6m": closes / closes.shift(126) - 1,
        "ret_12_1": closes.shift(21) / closes.shift(252) - 1,
        "vol_63": ret.rolling(63, min_periods=40).std(),
        "max_ret_21": ret.rolling(21, min_periods=15).max(),
        "min_ret_21": ret.rolling(21, min_periods=15).min(),
        "dist_52w_high": closes / closes.rolling(252, min_periods=200).max() - 1,
        "log_dollar_volume": np.log(dollar_volume.rolling(63, min_periods=40).median().where(lambda x: x > 0)),
    }
    return {k: v.reindex(ends) for k, v in feats.items()}


def new_members(members: pd.DataFrame, ends: pd.DatetimeIndex, days: int = 365) -> pd.DataFrame:
    """True where a stock is a member at the month end but was not `days` calendar days before."""
    m = members.astype(float)
    m = m.reindex(m.index.union(ends).union(ends - pd.Timedelta(days=days))).ffill().fillna(0) > 0.5
    now = m.reindex(ends)
    before = m.reindex(ends - pd.Timedelta(days=days))
    before.index = ends
    return now & ~before


def earnings_due(reaction: pd.DataFrame, ends: pd.DatetimeIndex, tickers, lag_months: int = 3) -> pd.DataFrame:
    """True for (month end, ticker) if the ticker had an earnings reaction day in the calendar month
    `lag_months` before the month that follows the month end (a release then is due again in the
    coming month). `reaction`: columns ticker, day. Uses only releases before the month end."""
    months = pd.DataFrame({"ticker": reaction["ticker"], "m": pd.DatetimeIndex(reaction["day"]).to_period("M")})
    had = set(zip(months["ticker"], months["m"]))
    out = pd.DataFrame(False, index=ends, columns=list(tickers))
    for e in ends:
        target = e.to_period("M") + 1 - lag_months
        out.loc[e] = [(t, target) in had for t in out.columns]
    return out


def frame(returns: pd.DataFrame, members: pd.DataFrame, features: dict) -> pd.DataFrame:
    """Long table (month end, ticker): the month's return, its rank groups and the features. Only stocks
    that are members at the month end and have a return."""
    m = members.astype(float)
    m = m.reindex(m.index.union(returns.index)).ffill().reindex(returns.index).fillna(0) > 0.5
    r = returns.where(m.reindex(columns=returns.columns, fill_value=False))
    df = r.stack().rename("ret").reset_index()
    df.columns = ["month", "ticker", "ret"]
    g = df.groupby("month")["ret"]
    n = g.transform("size")
    rank = g.rank(ascending=False, method="first")                  # 1 = best
    df["top10"] = rank <= 10
    df["bottom10"] = rank > n - 10
    pct = g.rank(pct=True, method="first")
    df["top_decile"] = pct > 0.9
    df["bottom_decile"] = pct <= 0.1
    df["excess"] = df["ret"] - g.transform("mean")
    key = pd.MultiIndex.from_frame(df[["month", "ticker"]])
    for name, panel in features.items():
        df[name] = panel.stack(future_stack=True).reindex(key).to_numpy()
    return df


def buckets(df: pd.DataFrame, col: str) -> pd.Series:
    """Q1..Q5 per month for a numeric feature (Q5 = highest), the value as text for a category."""
    x = df[col]
    if x.dtype == bool or x.dtype == object or isinstance(x.dtype, pd.CategoricalDtype):
        return x.astype("string")
    pct = x.groupby(df["month"]).rank(pct=True)
    q = np.ceil(pct * QUINTILES).clip(1, QUINTILES)
    return ("Q" + q.astype("Int64").astype("string")).where(x.notna())


def lift_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """For each bucket of `col`: rows, share in each group divided by the group's share among the rows
    with that feature (1 = average), mean excess return a month, and how many years the lift into the
    top decile exceeded the lift into the bottom decile."""
    b = buckets(df, col)
    d = df[b.notna()].assign(bucket=b[b.notna()], year=lambda x: x["month"].dt.year)
    base = d[list(GROUPS)].mean()
    out = d.groupby("bucket")[list(GROUPS)].mean() / base
    out.columns = [f"lift_{g}" for g in GROUPS]
    out.insert(0, "rows", d.groupby("bucket").size())
    out["asymmetry"] = out["lift_top_decile"] - out["lift_bottom_decile"]
    out["excess_pct_month"] = d.groupby("bucket")["excess"].mean() * 100
    yb = d.groupby("year")[["top_decile", "bottom_decile"]].mean()
    by = d.groupby(["bucket", "year"])[["top_decile", "bottom_decile"]].mean()
    asym = (by["top_decile"] / yb["top_decile"].reindex(by.index, level="year")
            - by["bottom_decile"] / yb["bottom_decile"].reindex(by.index, level="year"))
    out["years_top_gt_bottom"] = (asym > 0).groupby(level="bucket").sum()
    out["years"] = asym.groupby(level="bucket").size()
    out.insert(0, "feature", col)
    return out.reset_index()


def spread(df: pd.DataFrame, col: str, hi: str = "Q5", lo: str = "Q1") -> dict:
    """Monthly mean excess return of bucket `hi` minus `lo`, with its t-statistic over months."""
    b = buckets(df, col)
    m = df.assign(bucket=b).groupby(["month", "bucket"])["excess"].mean().unstack()
    if hi not in m or lo not in m:
        return {"feature": col, "months": 0, "spread_pct_month": np.nan, "t": np.nan}
    s = (m[hi] - m[lo]).dropna()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan
    return {"feature": col, "months": len(s), "spread_pct_month": s.mean() * 100, "t": t}


SIC_GROUPS = [((1000, 1499), "energy_mining"), ((2900, 2999), "energy_mining"), ((2800, 2899), "chemicals_pharma"),
              ((3570, 3579), "tech_hardware"), ((3600, 3699), "tech_hardware"), ((3800, 3899), "tech_hardware"),
              ((2000, 3999), "manufacturing"), ((4000, 4799), "transport"), ((4800, 4899), "telecom_media"),
              ((4900, 4999), "utilities"), ((5000, 5999), "trade"), ((6000, 6799), "finance_realestate"),
              ((7370, 7379), "software_services"), ((8000, 8099), "health_services"), ((7000, 8999), "services")]


def sic_group(sic) -> str:
    """Broad industry from a SIC code (first matching range; 'other' if none or missing)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "other"
    return next((name for (lo, hi), name in SIC_GROUPS if lo <= code <= hi), "other")
