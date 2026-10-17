"""Compare the Hugging Face daily bars (scripts/hf_daily.py) with the Quandl WIKI prices.

WIKI keeps the unadjusted close (`raw_close`) and the adjustment factor, so the HF
(unadjusted) bars can be checked directly: same company -> same raw close. Reports, per
year and overall: S&P 500 member coverage in HF and WIKI, share of days where HF and WIKI
agree on the raw close, ticker identity mismatches (reused tickers), how far the HF open,
high, low and volume are from WIKI's, and missing days.

    python scripts/hf_vs_wiki.py --hf data/hf_daily --wiki data/history_wiki --out data/hf_check
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.history import parse_members

TOL = 0.005          # raw closes within 0.5% = the same price


def load_hf(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def wiki_frame(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    f = d["close"] / d["raw_close"]                                # adjustment factor
    return pd.DataFrame({"raw_close": d["raw_close"], "raw_open": d["open"] / f,
                         "raw_high": d["high"] / f, "raw_low": d["low"] / f,
                         "wiki_volume": d["volume"] / (1 / f).where(f > 0)})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hf", default="data/hf_daily")
    p.add_argument("--wiki", default="data/history_wiki")
    p.add_argument("--out", default="data/hf_check")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hf = load_hf(Path(args.hf))
    lo, hi = hf["date"].min(), hf["date"].max()
    print(f"HF daily: {len(hf):,} ticker-days, {hf['ticker'].nunique():,} tickers, {lo.date()} to {hi.date()}")
    hf_by = {t: g.set_index("date").sort_index() for t, g in hf.groupby("ticker")}
    members = parse_members((Path(args.wiki) / "sp500_members.csv").read_text())
    members = members[(members["date"] >= lo) & (members["date"] <= hi)]
    wiki_files = {f.stem: f for f in (Path(args.wiki) / "wiki").glob("*.parquet")}

    rows, parts = [], []
    for t, f in wiki_files.items():
        w = wiki_frame(f)
        w = w[(w.index >= lo) & (w.index <= hi)].dropna(subset=["raw_close"])
        h = hf_by.get(t)
        if h is None or w.empty:
            rows.append(dict(ticker=t, wiki_days=len(w), hf_days=0, common=0))
            continue
        j = w.join(h, how="inner", rsuffix="_hf")
        if j.empty:
            rows.append(dict(ticker=t, wiki_days=len(w), hf_days=len(h), common=0))
            continue
        ratio = j["close"] / j["raw_close"]
        agree = (ratio - 1).abs() < TOL
        rows.append(dict(ticker=t, wiki_days=len(w), hf_days=len(h), common=len(j),
                         agree=float(agree.mean()), median_abs_diff_bp=float((ratio - 1).abs().median() * 1e4),
                         missing_in_hf=1 - len(j) / len(w)))
        ok = j[agree]
        parts.append(pd.DataFrame({
            "ticker": t, "year": ok.index.year,
            "open_bp": (ok["open"] / ok["raw_open"] - 1).abs() * 1e4,
            "high_bp": (ok["high"] / ok["raw_high"] - 1).abs() * 1e4,
            "low_bp": (ok["low"] / ok["raw_low"] - 1).abs() * 1e4,
            "close_bp": (ok["close"] / ok["raw_close"] - 1).abs() * 1e4,
            "x1600_bp": (ok["x1600"] / ok["raw_close"] - 1).abs() * 1e4,
            "vol_ratio": ok["volume"] / ok["wiki_volume"]}))
    t = pd.DataFrame(rows).set_index("ticker")
    t.to_csv(out / "per_ticker.csv")
    q = pd.concat(parts)
    q.to_csv(out / "per_day_agree.csv.gz", index=False)

    c = t[t["common"] > 0]
    print(f"\nWIKI tickers with days in the window: {(t['wiki_days'] > 0).sum()}; found in HF: {(t['hf_days'] > 0).sum()}; "
          f"with overlapping days: {len(c)}")
    print("ticker-level agreement (share of common days with raw close within 0.5%):")
    print(c["agree"].describe(percentiles=[.05, .1, .25, .5]).round(3).to_string())
    bad = c[c["agree"] < 0.9].sort_values("agree")
    print(f"\ntickers with <90% agreement (probably a different company or bad data): {len(bad)} of {len(c)}")
    print(bad.head(15)[["wiki_days", "hf_days", "common", "agree", "median_abs_diff_bp"]].round(3).to_string())
    good = c[c["agree"] >= 0.9]
    print(f"\nagreeing tickers: share of WIKI days missing in HF: mean {good['missing_in_hf'].mean():.3f}, "
          f"median {good['missing_in_hf'].median():.3f}")

    qq = q[q["ticker"].isin(good.index)]
    print("\nHF vs WIKI raw prices on agreeing days (median absolute difference, basis points):")
    print(qq[["open_bp", "high_bp", "low_bp", "close_bp", "x1600_bp"]].median().round(2).to_string())
    print("(close = last regular-session bar; x1600 = open of the 16:00 bar, NaN when absent)")
    print("volume ratio HF / WIKI (consolidated) median by year:")
    print(qq.groupby("year")["vol_ratio"].median().round(2).to_string())

    print("\nS&P 500 member coverage by year (members on a list date that year):")
    res = []
    for y, g in members.groupby(members["date"].dt.year):
        mem = set(g["ticker"])
        in_hf = {m for m in mem if m in hf_by and (hf_by[m].index.year == y).sum() >= 100}
        agree_ok = {m for m in in_hf if m in good.index}
        in_wiki = {m for m in mem if m in wiki_files}
        res.append(dict(year=y, members=len(mem), in_wiki=len(in_wiki), in_hf=len(in_hf), hf_agrees_with_wiki=len(agree_ok)))
    r = pd.DataFrame(res).set_index("year")
    print(r.to_string())
    r.to_csv(out / "coverage_by_year.csv")
    gone = [m for m in set(members["ticker"]) if m in wiki_files and m in c.index
            and wiki_frame(wiki_files[m]).index.max() < pd.Timestamp("2018-03-01")]
    gc = c.reindex(gone)
    print(f"\nWIKI members that stopped trading before Mar 2018 (delisted): {len(gone)}; of them in HF with agreeing prices: "
          f"{int((gc['agree'] >= 0.9).sum())}")
    print("examples not agreeing or missing:", sorted(gc[~(gc['agree'] >= 0.9)].index)[:20])


if __name__ == "__main__":
    main()
