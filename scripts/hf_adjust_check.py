"""How well can the unadjusted HF daily bars be adjusted?  (WIKI is the reference, 2008-2018.)

For every WIKI ticker whose HF prices agree with it (see hf_vs_wiki.py):
  A. WIKI events applied to HF raw prices: adjusted close return vs WIKI's adjusted return.
  B. Splits found in the HF prices alone (`detect_splits`): precision and recall against WIKI's
     splits, and the return error of an HF series adjusted with them, with and without
     the real dividends.

    python scripts/hf_adjust_check.py --hf data/hf_daily --wiki data/history_wiki --out data/hf_check
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.hfdata import adjust_frame, detect_splits
from hf_vs_wiki import load_hf


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hf", default="data/hf_daily")
    p.add_argument("--wiki", default="data/history_wiki")
    p.add_argument("--out", default="data/hf_check")
    args = p.parse_args(argv)
    out = Path(args.out)
    per = pd.read_csv(out / "per_ticker.csv", index_col=0)
    good = per[per["agree"] >= 0.9].index
    hf = load_hf(Path(args.hf))
    lo, hi = hf["date"].min(), hf["date"].max()
    hf_by = {t: g.set_index("date").sort_index() for t, g in hf.groupby("ticker") if t in set(good)}
    rows, splits_rows = [], []
    for t in good:
        w = pd.read_parquet(Path(args.wiki) / "wiki" / f"{t}.parquet")
        w = w[(w.index >= lo) & (w.index <= hi)]
        h = hf_by[t][["open", "high", "low", "close", "volume"]]
        h = h[h["close"] > 0]
        ev = w[["split", "dividend"]]
        ev = ev[(ev["split"] != 1) | (ev["dividend"] > 0)]
        j = w[["close"]].join(h[[]], how="inner").index          # days both have
        wr = w["close"].reindex(j).pct_change()
        res = dict(ticker=t, days=len(j), wiki_splits=int((ev["split"] != 1).sum()),
                   wiki_dividends=int((ev["dividend"] > 0).sum()))
        for name, adj in (("A_wiki_events", adjust_frame(h, ev)),
                          ("B_det_splits_real_div", adjust_frame(h, pd.DataFrame(
                              {"split": detect_splits(h), "dividend": ev["dividend"]}))),
                          ("B_det_splits_only", adjust_frame(h, pd.DataFrame(
                              {"split": detect_splits(h)})), )):
            ar = adj["close"].reindex(j).pct_change()
            d = (ar - wr).abs()
            res[f"{name}_median_bp"] = float(d.median() * 1e4)
            res[f"{name}_days_over_50bp"] = int((d > 0.005).sum())
            res[f"{name}_total_return_diff"] = float((1 + ar.dropna()).prod() / (1 + wr.dropna()).prod() - 1)
        det = detect_splits(h)
        wsp = ev.loc[ev["split"] != 1, "split"]
        pos = h.index.searchsorted(wsp.index)
        wdates = set(h.index[pos[pos < len(h)]])
        hit = set(det.index) & wdates
        res.update(det_splits=len(det), det_hit=len(hit), det_false=len(set(det.index) - wdates),
                   det_missed=len(wdates - set(det.index)))
        rows.append(res)
    r = pd.DataFrame(rows).set_index("ticker")
    r.to_csv(out / "adjust_check.csv")
    pd.set_option("display.width", 200)
    print(f"{len(r)} tickers, {r['days'].sum():,} days; WIKI splits {r['wiki_splits'].sum()}, dividends {r['wiki_dividends'].sum()}")
    for name in ("A_wiki_events", "B_det_splits_real_div", "B_det_splits_only"):
        print(f"\n{name}: daily adjusted-return error vs WIKI (basis points) ")
        print("  median of ticker medians:", round(r[f"{name}_median_bp"].median(), 2),
              "| tickers with any day over 50 bp:", int((r[f"{name}_days_over_50bp"] > 0).sum()),
              "| total days over 50 bp:", int(r[f"{name}_days_over_50bp"].sum()))
        td = r[f"{name}_total_return_diff"]
        print("  total-return difference over the whole period: median", round(td.median() * 100, 2),
              "% | 5th/95th percentile", round(td.quantile(.05) * 100, 2), "/", round(td.quantile(.95) * 100, 2), "%")
    print(f"\nsplit detection: found {r['det_splits'].sum()}, hit {r['det_hit'].sum()} of {r['wiki_splits'].sum()} WIKI splits "
          f"(recall {r['det_hit'].sum() / max(r['wiki_splits'].sum(), 1):.2f}), false {r['det_false'].sum()}, missed {r['det_missed'].sum()}")
    print("tickers with most false/missed detections:")
    print(r.assign(bad=r["det_false"] + r["det_missed"]).sort_values("bad", ascending=False).head(10)[
        ["wiki_splits", "det_splits", "det_hit", "det_false", "det_missed"]].to_string())


if __name__ == "__main__":
    main()
