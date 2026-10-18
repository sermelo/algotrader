"""A size-ranked universe built from our own daily data (for mid-cap tests of the factor strategies).

Index membership lists (Russell 1000, S&P 400) are not free, so each month the stocks are ranked by
trailing dollar volume (raw close x consolidated volume, median over the days of each month, averaged
over `window` months) and the ranks `lo` to `hi` are the universe. Dollar volume stands in for market
value: it is point-in-time, needs no membership file and includes delisted names. A list dated at a
month's last trading day uses only data up to that day.

Input: the monthly files of `scripts/hf_daily.py` (columns ticker, date, close, volume; prices NOT
adjusted, which is right for dollar volume). Output: the same `date,tickers` CSV as the S&P 500 members.
"""
import numpy as np
import pandas as pd


def month_stats(daily: pd.DataFrame) -> pd.DataFrame:
    """Per ticker for one month: median daily dollar volume, trading days, last raw close, and the
    month's last date (all tickers share the file's last date for the list date)."""
    d = daily.dropna(subset=["close", "volume"]).copy()
    d["dv"] = d["close"] * d["volume"]
    g = d.groupby("ticker")
    out = pd.DataFrame({"dv": g["dv"].median(), "days": g["dv"].size(), "price": g["close"].last()})
    out.attrs["date"] = d["date"].max()
    return out


def rank_members(stats: dict, lo: int = 501, hi: int = 1500, window: int = 3, min_price: float = 5.0,
                 min_days: int = 10, exclude=()) -> pd.DataFrame:
    """Long table (date, ticker) of the stocks ranked `lo`..`hi` by dollar volume at each month.

    `stats` maps the list date to `month_stats` output, in date order. A stock is ranked only if its
    last close is at least `min_price`, it traded at least `min_days` days in the month and it is not in
    `exclude` (ETFs and funds); its score is the mean over the last `window` months of its monthly
    median dollar volume (months it did not trade count as zero, so a new listing ranks low)."""
    dates = sorted(stats)
    if lo < 1 or hi < lo or window < 1:
        raise ValueError("require 1 <= lo <= hi, window >= 1")
    dv = pd.DataFrame({d: stats[d]["dv"] for d in dates}).fillna(0.0).T
    rows = []
    for i, d in enumerate(dates):
        cur = stats[d]
        ok = (cur["price"] >= min_price) & (cur["days"] >= min_days) & ~cur.index.isin(list(exclude))
        score = dv.iloc[max(0, i - window + 1):i + 1].mean()[cur.index[ok]].sort_values(ascending=False)
        for t in score.index[lo - 1:hi]:
            rows.append((d, t))
    return pd.DataFrame(rows, columns=["date", "ticker"])


def to_csv_text(members: pd.DataFrame) -> str:
    """The `date,tickers` CSV (tickers comma-joined, sorted) that `history.parse_members` reads."""
    lines = ["date,tickers"]
    for d, g in members.groupby("date"):
        lines.append(f'{pd.Timestamp(d).date()},"{",".join(sorted(g["ticker"]))}"')
    return "\n".join(lines) + "\n"
