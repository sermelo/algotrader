"""Where could an intraday edge be? Scans of the minute-feature dataset (roadmap S14).

Reads the monthly files written by `scripts/hf_minute_features.py` and writes, in `--out`:
`conditional.csv` (move statistics by bin of each condition, `algotrader.minutescan.BinTable`),
`signal_scan.csv` (rank correlation and top/bottom picks of each feature against each forward return,
`SignalScan`) and `summary.md` (the tables worth reading).

    python scripts/minute_scan.py --data data/hf_minute --out data/minute_scan
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.minutescan import CONDITIONS, DISCRETE, THRESHOLDS, BinTable, SignalScan, derived, quantile_edges

COST = 0.0014                                       # the 667-euro slot on Tiered, for the "typical move / cost" columns


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/hf_minute")
    p.add_argument("--out", default="data/minute_scan")
    p.add_argument("--vol-rank", type=float, default=None, help="keep only the stocks at or above this rank (0-1) of 30-minute volatility within each decision")
    a = p.parse_args(argv)
    files = sorted(f for f in Path(a.data).glob("*.parquet") if not f.name.startswith("carry"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = pd.concat([pd.read_parquet(f) for f in files[::6]], ignore_index=True)
    bins = {n: BinTable(n, None if n in DISCRETE else quantile_edges(CONDITIONS[n](sample).to_numpy(), 10)) for n in CONDITIONS}
    scan = SignalScan()
    del sample
    t0 = time.time()
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        if a.vol_rank:
            df = df[df.groupby(["date", "minute"], sort=False)["vol30"].rank(pct=True) >= a.vol_rank]
        df = derived(df)
        scan.add(df)
        for b in bins.values():
            b.add(df)
        if i % 24 == 0:
            print(f.stem, f"{time.time() - t0:.0f} s", flush=True)
    cond = pd.concat([b.table() for b in bins.values()], ignore_index=True)
    sig = scan.result()
    cond.to_csv(out / "conditional.csv", index=False, float_format="%.6g")
    sig.to_csv(out / "signal_scan.csv", index=False, float_format="%.6g")

    L = ["# Minute-feature scans (2008-2026, S&P 500 members, decisions at 10:00, 10:30 ... 15:00" + (f"; only the top {1 - a.vol_rank:.0%} of stocks by 30-minute volatility at each decision" if a.vol_rank else "") + ")", "",
         f"## Where are the big moves? Typical move (mean absolute return) as a multiple of a {COST * 100:.2f}% cost, lowest to highest tenth of each condition", ""]
    for lab in ("fwd30", "fwd90", "fwd240"):
        L += [f"### Horizon {lab[3:]} minutes", "", "| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |", "|---|---|---|---|---|"]
        for n in CONDITIONS:
            if n in DISCRETE:
                continue
            t = cond[(cond["condition"] == n) & (cond["label"] == lab)].sort_values("bin")
            if len(t) < 3:
                continue
            thr = [c for c in THRESHOLDS if abs(c - COST) < 1e-9][0]
            L.append(f"| {n} | {t['mean_abs'].iloc[0] / COST:.1f}x | {t['mean_abs'].iloc[len(t) // 2] / COST:.1f}x | {t['mean_abs'].iloc[-1] / COST:.1f}x ({t['bin_value'].iloc[-1]:.3g}) | {t[f'up_{thr}'].iloc[-1] * 100:.0f}% |")
        L.append("")
    L += ["## Time of day (decision minute after 09:30): mean return and typical move", "", "| Minute | Horizon 30: mean | Horizon 30: typical move | Horizon 90: mean | Horizon 90: typical move | To the close: mean |", "|---|---|---|---|---|---|"]
    tm = cond[cond["condition"] == "minute"]
    for m in sorted(tm["bin_value"].unique()):
        g = tm[tm["bin_value"] == m].set_index("label")
        row = lambda lab, k: g.loc[lab, k] * 100 if lab in g.index else float("nan")
        L.append(f"| {int(m)} | {row('fwd30', 'mean'):+.3f}% | {row('fwd30', 'mean_abs'):.3f}% | {row('fwd90', 'mean'):+.3f}% | {row('fwd90', 'mean_abs'):.3f}% | {row('fwd_close', 'mean'):+.3f}% |")
    n_tests = len(sig)
    L += ["", f"## Which features rank stocks by later return? ({n_tests} feature x horizon tests; a |t| above about 3.3 is needed for 5% after testing this many)", "",
          "IC = average rank correlation between the feature and the forward return across stocks at each decision; t over the days; years positive = calendar years with a positive average IC; excess returns in basis points against the average stock of the same decision (top/bottom tenth, and the single top/bottom stock).", "",
          "| Feature | Horizon | IC | t | Years positive | Top tenth excess | Top stock excess | Top stock raw return | Bottom stock excess |", "|---|---|---|---|---|---|---|---|---|"]
    top = sig.reindex(sig["ic_t"].abs().sort_values(ascending=False).index).head(25)
    for _, r in top.iterrows():
        L.append(f"| {r['feature']} | {r['label'][3:]} | {r['ic']:+.4f} | {r['ic_t']:+.1f} | {int(r['years_positive'])}/{int(r['years'])} | {r['top10_bp']:+.1f} bp | {r['top1_bp']:+.1f} bp | {r['top1raw_bp']:+.1f} bp | {r['bot1_bp']:+.1f} bp |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
