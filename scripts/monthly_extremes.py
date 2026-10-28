"""Biggest monthly winners and losers of the S&P 500 members, 2009 to March 2026 (TASKS 11, step 1).

Descriptive only (no trading rule, no configuration counted): writes results/extremes/ with the lift of each
feature known at the previous month end into the top/bottom 10 and deciles (`lift.csv`), the Q5 - Q1 excess
return (`spread.csv`), every month's top and bottom 10 with their features (`extremes.csv`) and `summary.md`.
Ends in March 2026: after that 166 members' prices stop (README, Limitations) and would look like delistings.

    uv run python scripts/monthly_extremes.py      # on the VPS (data/history, data/edgar)
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algotrader import extremes, longrun                                    # noqa: E402
from algotrader.history import parse_members                                # noqa: E402
from algotrader.pead import reaction_days                                   # noqa: E402

OUT = Path("results/extremes")
FIRST, LAST = "2009-01-01", "2026-03-31"
EDGAR = longrun.EDGAR
NUMERIC = ["ret_1m", "ret_6m", "ret_12_1", "vol_63", "max_ret_21", "min_ret_21", "dist_52w_high",
           "log_dollar_volume", "book_to_market", "gross_profitability", "asset_growth", "operating_profitability"]
CATEGORIES = ["earnings_due", "new_member", "industry"]


def main():
    frames = longrun.load_frames()
    members = longrun.membership_matrix(parse_members((longrun.HISTORY / "sp500_members.csv").read_text()))
    stocks = {t: f for t, f in frames.items() if t != longrun.BENCHMARK}
    closes = pd.DataFrame({t: f["close"] for t, f in stocks.items()})
    dollar_volume = pd.DataFrame({t: f["close"] * f["volume"] for t, f in stocks.items()})
    ends = extremes.month_ends(closes.index)
    ends = ends[(ends >= pd.Timestamp(FIRST) - pd.offsets.MonthEnd(1)) & (ends <= pd.Timestamp(LAST))]
    returns = extremes.monthly_returns(closes, ends)

    feats = extremes.price_features(closes, dollar_volume, ends)
    for name, panel in longrun.fundamental_panels(frames)["panels"].items():
        feats[name] = panel.reindex(index=ends, columns=closes.columns)
    cik = pd.read_csv(EDGAR / "cik_map.csv", index_col=0)
    tick_of = {}
    for t, c in cik["cik"].dropna().items():
        if t in closes.columns:
            tick_of.setdefault(int(c), []).append(t)
    earn = pd.read_parquet(EDGAR / "earnings.parquet")
    earn = earn[earn["cik"].isin(tick_of)].copy()
    earn["day"] = reaction_days(earn, closes.index)
    earn = earn.dropna(subset=["day"])
    reaction = pd.DataFrame([(t, d) for c, d in zip(earn["cik"], earn["day"]) for t in tick_of[int(c)]],
                            columns=["ticker", "day"])
    has_earn = set(reaction["ticker"])
    due = extremes.earnings_due(reaction, ends, closes.columns)
    feats["earnings_due"] = due.where(pd.DataFrame({t: t in has_earn for t in due.columns}, index=due.index))
    feats["new_member"] = extremes.new_members(members, ends).reindex(columns=closes.columns, fill_value=False)
    sic = pd.read_parquet(EDGAR / "companies.parquet", columns=["cik", "sic"]).set_index("cik")["sic"]
    industry = {t: extremes.sic_group(sic.get(int(c))) for t, c in cik["cik"].dropna().items()}
    feats["industry"] = pd.DataFrame({t: industry.get(t, "other") for t in closes.columns}, index=ends)

    df = extremes.frame(returns, members, feats)
    df = df[df["month"] < pd.Timestamp(LAST)]
    for c in ("earnings_due", "new_member"):
        df[c] = df[c].map({True: "yes", False: "no"})
    OUT.mkdir(parents=True, exist_ok=True)
    lift = pd.concat([extremes.lift_table(df, c) for c in NUMERIC + CATEGORIES], ignore_index=True)
    lift.to_csv(OUT / "lift.csv", index=False, float_format="%.4f")
    spread = pd.DataFrame([extremes.spread(df, c) for c in NUMERIC]
                          + [extremes.spread(df, c, "yes", "no") for c in ("earnings_due", "new_member")])
    spread.to_csv(OUT / "spread.csv", index=False, float_format="%.4f")
    tails = df[df["top10"] | df["bottom10"]].sort_values(["month", "ret"], ascending=[True, False])
    tails.to_csv(OUT / "extremes.csv", index=False, float_format="%.5g")
    write_summary(df, lift, spread)
    code = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in ("algotrader/extremes.py", "scripts/monthly_extremes.py")}
    (OUT / "manifest.json").write_text(json.dumps({"months": int(df["month"].nunique()), "rows": len(df),
                                                   "first": str(df["month"].min().date()),
                                                   "last": str(df["month"].max().date()),
                                                   "data": longrun.data_fingerprint(), "code": code}, indent=2))


def write_summary(df, lift, spread):
    months = df["month"].nunique()
    lines = [f"# Biggest monthly winners and losers (S&P 500 members)", "",
             f"{months} months (start month ends {df['month'].min().date()} to {df['month'].max().date()}), "
             f"{len(df):,} stock-months. Features are known at the month end before the month. Lift = share of the "
             "bucket's stock-months in the group / the group's share among stock-months with that feature "
             "(1 = average). Asymmetry = top-decile lift - bottom-decile lift; years = years in which the "
             "bucket's top-decile lift exceeded its bottom-decile lift. Excess = return minus the members' mean "
             "that month. Descriptive only: no rule, no costs. Generated by `scripts/monthly_extremes.py`.", "",
             "## Lift by bucket", "",
             "| Feature | Bucket | Rows | Top 10 | Bottom 10 | Top decile | Bottom decile | Asymmetry | "
             "Years top > bottom | Excess %/month |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in lift.iterrows():
        lines.append(f"| {r['feature']} | {r['bucket']} | {r['rows']:,} | {r['lift_top10']:.2f} | "
                     f"{r['lift_bottom10']:.2f} | {r['lift_top_decile']:.2f} | {r['lift_bottom_decile']:.2f} | "
                     f"{r['asymmetry']:+.2f} | {r['years_top_gt_bottom']}/{r['years']} | {r['excess_pct_month']:+.2f} |")
    lines += ["", "## Highest minus lowest bucket (excess return a month)", "",
              "| Feature | Months | Spread %/month | t |", "|---|---|---|---|"]
    for _, r in spread.iterrows():
        lines.append(f"| {r['feature']} | {r['months']} | {r['spread_pct_month']:+.2f} | {r['t']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
