"""Do the top strategies do better in some weeks or months of the year, or in earnings season? (sergio, 2026-10-29)

Descriptive check on the committed daily returns (base costs, 2012-01-03 to 2026-10-16) of the two paper books and the
other top long-history configurations, SPY and cash (Fama-French risk-free rate, to 2026-08-31, then 0):

1. Week of the year (ISO week of the week's last session; week 53 counted as 52) and calendar month: mean return,
   mean return over SPY, its t-statistic across years, years positive.
2. Is the pattern more than chance? (a) Shuffle test: the spread (standard deviation) of the 52 weekly or 12 monthly
   means of the return over SPY against the same with the week/month labels shuffled within each year (2,000 times).
   (b) Split halves: rank correlation of the weekly/monthly means over SPY in 2012-2018 and 2019-2026.
3. Earnings season: days with many S&P 500 earnings releases (8-K item 2.02 reaction days of the mapped members; a
   day is in season if its centred 10-session count of releases is in the top 40% of its year), mean daily return over
   SPY in and out of season, the difference by year and its t-statistic.
4. Holdout for the strongest month: the concentration test also ran 2008-2011 (before the 2012 start used above),
   so the best month over SPY found above is checked there for momentum (top 3, 5, 20, monthly).
5. A rule a trader could follow: each January, for each calendar month, hold the option (any of the above, SPY or
   cash) with the highest mean return in that month over all earlier years (at least 3), for that month of the year;
   0.10% of the portfolio is charged at each switch. Against holding each option all the time.

Results in results/calendar/ (summary.md, by_week.csv, by_month.csv). No strategy or configuration is added to the
trials count; step 5 is one extra rule and is reported against the 1,226 tries.

    uv run python scripts/calendar_check.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algotrader import french                                               # noqa: E402
from algotrader.pead import reaction_days                                   # noqa: E402
from algotrader.stats import returns_stats                                  # noqa: E402

OUT = Path("results/calendar")
START = "2012-01-01"
SEASON_TOP = 0.4
SWITCH_COST = 0.001
SHUFFLES = 2000
SERIES = {                                     # label: (results folder, returns column)
    "op_profitability top 10 (paper)": ("long_conc", "op_profitability|top10|base"),
    "momentum top 5 weekly (paper)": ("long_rebal", "momentum|top5|W|base"),
    "momentum top 3": ("long_conc", "momentum|top3|base"),
    "asset_growth top 5": ("long_conc", "asset_growth|top5|base"),
    "low_vol top 20": ("long_conc", "low_vol|top20|base"),
    "SPY": ("long_conc", "SPY"),
}


def load_returns() -> pd.DataFrame:
    cols = {}
    for label, (folder, col) in SERIES.items():
        r = pd.read_csv(f"results/{folder}/returns.csv", index_col=0, parse_dates=True)[col]
        cols[label] = r
    df = pd.DataFrame(cols).dropna()
    df = df[df.index >= START]
    rf = french.load()["RF"].reindex(df.index).fillna(0.0)
    df["cash"] = rf
    return df


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = (1 + df).groupby(df.index.to_period("W-FRI")).prod() - 1
    end = df.index.to_series().groupby(df.index.to_period("W-FRI")).max()
    w.index = pd.DatetimeIndex(end.to_numpy())
    return w


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    return (1 + df).groupby(df.index.to_period("M")).prod().sub(1).set_axis(
        df.index.to_series().groupby(df.index.to_period("M")).max().to_numpy())


def group_table(per: pd.DataFrame, key: pd.Series, name: str) -> pd.DataFrame:
    """Per strategy and group: mean return, mean over SPY, its t across years, years positive."""
    rows = []
    years = per.index.year
    for s in per.columns:
        for g, idx in per.groupby(key).groups.items():
            # one value per year: the group's compounded return in that year (over SPY: difference of those)
            yr = (1 + per.loc[idx, s]).groupby(years[per.index.isin(idx)]).prod() - 1
            yr_spy = (1 + per.loc[idx, "SPY"]).groupby(years[per.index.isin(idx)]).prod() - 1
            d = yr - yr_spy
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
            rows.append({"strategy": s, name: g, "years": len(d), "mean_pct": 100 * yr.mean(),
                         "over_spy_pct": 100 * d.mean(), "t": t, "years_positive": int((yr > 0).sum()),
                         "years_beat_spy": int((d > 0).sum())})
    return pd.DataFrame(rows)


def shuffle_test(per: pd.DataFrame, key: pd.Series, s: str, rng) -> tuple:
    """(spread of the group means of the return over SPY, share of shuffles with a larger spread)."""
    ex = (per[s] - per["SPY"]).to_numpy()
    k = key.to_numpy()
    years = per.index.year.to_numpy()
    groups = np.unique(k)

    def spread(labels):
        return np.std([ex[labels == g].mean() for g in groups])

    real = spread(k)
    blocks = [np.where(years == y)[0] for y in np.unique(years)]
    larger = 0
    for _ in range(SHUFFLES):
        lab = k.copy()
        for b in blocks:
            lab[b] = rng.permutation(lab[b])
        larger += spread(lab) >= real
    return real, larger / SHUFFLES


def split_half(per: pd.DataFrame, key: pd.Series, s: str) -> float:
    ex = per[s] - per["SPY"]
    first = ex[per.index.year <= 2018].groupby(key[per.index.year <= 2018]).mean()
    second = ex[per.index.year >= 2019].groupby(key[per.index.year >= 2019]).mean()
    both = pd.concat([first, second], axis=1).dropna()
    return both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank())


def season_days(index: pd.DatetimeIndex) -> pd.Series:
    earn = pd.read_parquet("data/edgar/earnings.parquet")
    cik = pd.read_csv("data/edgar/cik_map.csv", index_col=0)["cik"].dropna().astype(int)
    earn = earn[earn["cik"].isin(set(cik))].reset_index(drop=True)
    days = reaction_days(earn, index).dropna()
    count = days.value_counts().reindex(index, fill_value=0).sort_index()
    smooth = count.rolling(10, center=True, min_periods=1).sum()
    cut = smooth.groupby(smooth.index.year).transform(lambda x: x.quantile(1 - SEASON_TOP))
    return smooth > cut


def season_table(df: pd.DataFrame, season: pd.Series) -> pd.DataFrame:
    rows = []
    years = df.index.year
    for s in df.columns:
        ex = df[s] - df["SPY"]
        by = pd.DataFrame({"ex": ex, "in": season, "y": years})
        per_year = by.groupby(["y", "in"])["ex"].mean().unstack()
        d = (per_year[True] - per_year[False]) * 252
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else np.nan
        rows.append({"strategy": s, "in_season_over_spy_pct_yr": 100 * 252 * ex[season].mean(),
                     "off_season_over_spy_pct_yr": 100 * 252 * ex[~season].mean(),
                     "in_season_pct_yr": 100 * 252 * df.loc[season, s].mean(),
                     "off_season_pct_yr": 100 * 252 * df.loc[~season, s].mean(),
                     "difference_pct_yr": 100 * d.mean(), "t": t, "years_in_better": int((d > 0).sum()),
                     "years": len(d)})
    return pd.DataFrame(rows)


def holdout(best_month: int) -> pd.DataFrame:
    """Momentum minus SPY in `best_month` of 2008-2011, from the concentration test's returns."""
    c = pd.read_csv("results/long_conc/returns.csv", index_col=0, parse_dates=True)
    c = c[c.index < START]
    m = (1 + c).groupby(c.index.to_period("M")).prod() - 1
    m = m[m.index.month == best_month]
    rows = {col.split("|")[1]: 100 * (m[col] - m["SPY"]) for col in
            ("momentum|top3|base", "momentum|top5|base", "momentum|top20|base")}
    return pd.DataFrame(rows).set_axis(m.index.year)


def month_switch(df: pd.DataFrame, min_years=3) -> tuple:
    """Daily returns of the walk-forward month rule and the choices it made."""
    month = df.index.month
    year = df.index.year
    out = pd.Series(np.nan, index=df.index)
    choices = {}
    prev = None
    for y in sorted(set(year)):
        past = df[year < y]
        if past.index.year.nunique() < min_years:
            continue
        mret = (1 + past).groupby([past.index.year, past.index.month]).prod() - 1
        best = mret.groupby(level=1).mean().idxmax(axis=1)
        for m in range(1, 13):
            idx = (year == y) & (month == m)
            if not idx.any():
                continue
            pick = best.get(m)
            r = df.loc[idx, pick].copy()
            if prev is not None and pick != prev:
                r.iloc[0] -= SWITCH_COST
            out[idx] = r
            choices[(y, m)] = pick
            prev = pick
    return out.dropna(), choices


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_returns()
    wk = weekly(df)
    week_no = pd.Series(np.minimum(wk.index.isocalendar().week.to_numpy(), 52), index=wk.index)
    mo = monthly(df)
    month_no = pd.Series(mo.index.month, index=mo.index)
    rng = np.random.default_rng(0)
    strategies = [s for s in df.columns if s not in ("SPY", "cash")]

    by_week = group_table(wk, week_no, "week")
    by_month = group_table(mo, month_no, "month")
    by_week.to_csv(OUT / "by_week.csv", index=False, float_format="%.4f")
    by_month.to_csv(OUT / "by_month.csv", index=False, float_format="%.4f")
    tests = []
    for s in strategies + ["cash"]:
        sw, pw = shuffle_test(wk, week_no, s, rng)
        sm, pm = shuffle_test(mo, month_no, s, rng)
        tests.append({"strategy": s, "week_shuffle_p": pw, "week_split_corr": split_half(wk, week_no, s),
                      "month_shuffle_p": pm, "month_split_corr": split_half(mo, month_no, s)})
    tests = pd.DataFrame(tests)
    season = season_days(df.index)
    seas = season_table(df, season)
    top = by_month[by_month["strategy"] == "momentum top 5 weekly (paper)"].nlargest(1, "t").iloc[0]
    hold = holdout(int(top["month"]))
    rule, choices = month_switch(df)
    common = df.loc[rule.index]
    compare = [{"option": "month rule (walk-forward)", **returns_stats(rule)}]
    compare += [{"option": s, **returns_stats(common[s])} for s in df.columns]
    compare = pd.DataFrame(compare)
    write_summary(df, by_week, by_month, tests, seas, season, compare, choices, rule, top, hold)
    code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("scripts/calendar_check.py",)}
    inputs = {f"results/{f}/returns.csv": hashlib.sha256(Path(f"results/{f}/returns.csv").read_bytes()).hexdigest()
              for f in sorted({v[0] for v in SERIES.values()})}
    (OUT / "manifest.json").write_text(json.dumps({"code": code, "inputs": inputs, "first": str(df.index.min().date()),
                                                   "last": str(df.index.max().date())}, indent=2))


def pct(x):
    return f"{x:+.1f}%"


def write_summary(df, by_week, by_month, tests, seas, season, compare, choices, rule, top, hold):
    L = ["# Calendar check: week of year, month, earnings season", "",
         f"Daily returns at base costs, {df.index.min().date()} to {df.index.max().date()} ({df.index.year.nunique()} "
         "calendar years, 2026 partial). Generated by `scripts/calendar_check.py`; descriptive (see its docstring). "
         "Over SPY = the strategy's compounded return in that week/month of a year minus SPY's, averaged over years; "
         "t across years.", "",
         "## Is there a calendar pattern beyond chance?", "",
         "Shuffle p: share of 2,000 within-year shuffles of the week (month) labels whose spread of the group means "
         "over SPY is at least the real one (small = a real pattern). Split corr: rank correlation of the group means "
         "over SPY in 2012-2018 and in 2019-2026 (near 0 = the pattern does not repeat).", "",
         "| Strategy | Week shuffle p | Week split corr | Month shuffle p | Month split corr |", "|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        L.append(f"| {r['strategy']} | {r['week_shuffle_p']:.2f} | {r['week_split_corr']:+.2f} | "
                 f"{r['month_shuffle_p']:.2f} | {r['month_split_corr']:+.2f} |")
    L += ["", "## By month (mean of each year's return in that month)", ""]
    piv = by_month.pivot(index="month", columns="strategy", values="mean_pct")[list(df.columns)]
    L += ["| Month | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
    for m, r in piv.iterrows():
        L.append(f"| {m} | " + " | ".join(pct(v) for v in r) + " |")
    L += ["", "Over SPY by month (t across years):", ""]
    ov = by_month[by_month["strategy"].isin([c for c in df.columns if c != "SPY"])]
    piv2 = ov.pivot(index="month", columns="strategy", values="over_spy_pct")
    pt = ov.pivot(index="month", columns="strategy", values="t")
    cols = [c for c in df.columns if c != "SPY"]
    L += ["| Month | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in piv2.index:
        L.append(f"| {m} | " + " | ".join(f"{piv2.loc[m, c]:+.1f}% ({pt.loc[m, c]:.1f})" for c in cols) + " |")
    L += ["", "## Weeks with the largest gap to SPY", "",
          "With 52 weeks per strategy, about 2-3 weeks reach |t| > 2 by chance alone.", "",
          "| Strategy | Week | Mean | Over SPY | t | Years beat SPY |", "|---|---|---|---|---|---|"]
    for s in [c for c in df.columns if c not in ("SPY", "cash")]:
        sub = by_week[by_week["strategy"] == s].dropna(subset=["t"])
        for _, r in pd.concat([sub.nlargest(2, "t"), sub.nsmallest(2, "t")]).iterrows():
            L.append(f"| {s} | {int(r['week'])} | {pct(r['mean_pct'])} | {pct(r['over_spy_pct'])} | {r['t']:.2f} | "
                     f"{r['years_beat_spy']}/{r['years']} |")
    n_t2 = int((by_week.loc[~by_week["strategy"].isin(["SPY"]), "t"].abs() > 2).sum())
    n_all = int(by_week.loc[~by_week["strategy"].isin(["SPY"]), "t"].notna().sum())
    L += ["", f"Weeks with |t| > 2 over SPY: {n_t2} of {n_all} (about {0.05 * n_all:.0f} expected by chance if they were "
          "independent; they are not: the two momentum versions share most holdings).",
          "", "## Earnings season", "",
          f"In season: {season.mean():.0%} of trading days (centred 10-session count of S&P 500 earnings releases in "
          "the top 40% of its year). Annualised mean daily returns; difference = in minus out of season over SPY, by "
          "year.", "",
          "| Strategy | In season | Off season | Over SPY in | Over SPY off | Difference (t) | Years in better |",
          "|---|---|---|---|---|---|---|"]
    for _, r in seas.iterrows():
        L.append(f"| {r['strategy']} | {pct(r['in_season_pct_yr'])} | {pct(r['off_season_pct_yr'])} | "
                 f"{pct(r['in_season_over_spy_pct_yr'])} | {pct(r['off_season_over_spy_pct_yr'])} | "
                 f"{pct(r['difference_pct_yr'])} ({r['t']:.2f}) | {r['years_in_better']}/{r['years']} |")
    L += ["", f"## Holdout: month {int(top['month'])} for momentum in 2008-2011", "",
          f"The strongest month over SPY above: momentum top 5 weekly, month {int(top['month'])}, "
          f"{pct(top['over_spy_pct'])} a year over SPY (t {top['t']:.2f}, beat SPY {int(top['years_beat_spy'])} of "
          f"{int(top['years'])} years). The same month in the four years before (monthly momentum, over SPY):", "",
          "| Year | " + " | ".join(hold.columns) + " |", "|---|" + "---|" * len(hold.columns)]
    for y, r in hold.iterrows():
        L.append(f"| {y} | " + " | ".join(pct(v) for v in r) + " |")
    L += ["", "## A rule a trader could follow: best option for each month, chosen on earlier years", "",
          f"From {rule.index.min().date()} (3 earlier years needed); {SWITCH_COST:.2%} charged per switch. "
          f"{sum(1 for i, (k, v) in enumerate(sorted(choices.items())) if i and v != sorted(choices.items())[i - 1][1])} "
          "switches.", "",
          "| Option | CAGR | Per month | Sharpe | Max drawdown |", "|---|---|---|---|---|"]
    for _, r in compare.iterrows():
        sharpe = "-" if r["option"] == "cash" else f"{r['sharpe']:.2f}"
        L.append(f"| {r['option']} | {r['cagr']:+.1%} | {(1 + r['cagr']) ** (1 / 12) - 1:+.2%} | {sharpe} | "
                 f"{r['max_drawdown']:.1%} |")
    picks = pd.Series(choices).value_counts()
    L += ["", "Months held by the rule: " + ", ".join(f"{k} {v}" for k, v in picks.items()) + "."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
