"""Compare the adjusted HF series (history/hf) with Tiingo after Mar 2018, per status.

    python scripts/hf_audit_tiingo.py --history data/history
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.history import tiingo_symbol, wiki_symbol


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="data/history")
    p.add_argument("--from-date", default="2018-04-01")
    a = p.parse_args(argv)
    h = Path(a.history)
    st = pd.read_csv(h / "hf_status.csv", index_col=0)
    rows = []
    for t, r in st.iterrows():
        if r["status"] not in ("ok", "unverified"):
            continue
        tp = h / "tiingo" / f"{tiingo_symbol(t)}.parquet"
        if not tp.exists():
            continue
        hf = pd.read_parquet(h / "hf" / f"{wiki_symbol(t)}.parquet")
        ti = pd.read_parquet(tp)
        hf, ti = hf[hf.index >= a.from_date], ti[ti.index >= a.from_date]
        j = hf.index.intersection(ti.index)
        if len(j) < 60:
            continue
        agree = float(((hf.loc[j, "raw_close"] / ti.loc[j, "raw_close"] - 1).abs() < 0.005).mean())
        rh, rt = hf.loc[j, "close"].pct_change(), ti.loc[j, "close"].pct_change()
        d = (rh - rt).abs()
        years = len(j) / 252
        tot = (1 + rh.dropna()).prod() / (1 + rt.dropna()).prod() - 1
        rows.append(dict(ticker=t, status=r["status"], days=len(j), raw_agree=agree,
                         median_ret_diff_bp=float(d.median() * 1e4), days_over_50bp=int((d > 0.005).sum()),
                         total_return_diff=float(tot), per_year=float((1 + tot) ** (1 / years) - 1),
                         tiingo_dividends=int((ti["dividend"] > 0).sum()), tiingo_splits=int((ti["split"] != 1).sum())))
    r = pd.DataFrame(rows).set_index("ticker")
    r.to_csv(h.parent / "hf_audit_tiingo.csv")
    print(f"{len(r)} tickers with both HF and Tiingo after {a.from_date}")
    for s, g in r.groupby("status"):
        bad = g[g["raw_agree"] < 0.9]
        print(f"\n[{s}] {len(g)} tickers; raw close agreement median {g['raw_agree'].median():.3f}; "
              f"<90% agreement (other company or missed split): {len(bad)}")
        ok = g[g["raw_agree"] >= 0.9]
        print(f"  agreeing: median daily return difference {ok['median_ret_diff_bp'].median():.2f} bp; "
              f"return difference per year (HF - Tiingo, missing dividends): median {ok['per_year'].median():+.2%}, "
              f"5th/95th pct {ok['per_year'].quantile(.05):+.2%} / {ok['per_year'].quantile(.95):+.2%}")
        if len(bad):
            print("  examples:", ", ".join(bad.sort_values("raw_agree").index[:12]))


if __name__ == "__main__":
    main()
