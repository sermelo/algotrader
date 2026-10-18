"""Build the size-ranked universe CSV from the monthly daily files.

    python scripts/size_universe.py data/hf_daily data/history/size_members.csv [--lo 501 --hi 1500]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from algotrader.sizeuniverse import month_stats, rank_members, to_csv_text

p = argparse.ArgumentParser()
p.add_argument("daily_dir")
p.add_argument("out")
p.add_argument("--lo", type=int, default=501)
p.add_argument("--hi", type=int, default=1500)
p.add_argument("--window", type=int, default=3)
p.add_argument("--min-price", type=float, default=5.0)
p.add_argument("--ex-members", default=None, help="drop S&P 500 members (sp500_members.csv): an ex-S&P 500 universe")
p.add_argument("--funds", default=None, help="SEC company_tickers_mf.json: tickers to leave out (ETFs, funds)")
a = p.parse_args()

stats = {}
for f in sorted(Path(a.daily_dir).glob("*.parquet")):
    s = month_stats(pd.read_parquet(f, columns=["ticker", "date", "close", "volume"]))
    stats[s.attrs["date"]] = s
    print(f.stem, len(s), flush=True)
funds = {r[3] for r in json.loads(Path(a.funds).read_text())["data"]} if a.funds else set()
print(len(funds), "fund tickers excluded")
members = rank_members(stats, a.lo, a.hi, a.window, a.min_price, exclude=funds)
if a.ex_members:
    from algotrader.history import parse_members
    sp = parse_members(Path(a.ex_members).read_text())
    lists = {d: set(g["ticker"]) for d, g in sp.groupby("date")}
    starts = sorted(lists)
    def sp_on(d):
        k = max((x for x in starts if x <= d), default=None)
        return lists[k] if k is not None else set()
    members = members[[t not in sp_on(d) for d, t in zip(members["date"], members["ticker"])]]
Path(a.out).write_text(to_csv_text(members))
print(len(stats), "months;", members.groupby("date").size().describe()[["min", "mean", "max"]].round(0).to_dict())
