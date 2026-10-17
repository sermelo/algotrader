"""Do simple stock-picking rules beat the cost of trading? (roadmap S14)

For every combination of a score (recent return, gap, VWAP distance ...), a direction (the lowest or the highest
stocks), a volatility filter, a number of picks (1 = one 2,000-euro position, 3 = three slots) and every decision
minute and horizon, the mean return of the picks, the excess over the average filtered stock and the result after
the round-trip cost (`algotrader.minutescan.RuleScan`). Writes `rule_check.csv` and `summary.md` in `--out`.

    python scripts/minute_rule_check.py --data data/hf_minute --out data/minute_rules
"""
import argparse
import sys
import time
from math import log, sqrt
from pathlib import Path

import pandas as pd

from algotrader.minutescan import RuleScan

SCORES = ["ret5", "ret30", "ret_open", "gap", "vwap_dist", "range_pos", "rvol30", "prev_ret"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/hf_minute")
    p.add_argument("--out", default="data/minute_rules")
    a = p.parse_args(argv)
    files = sorted(f for f in Path(a.data).glob("*.parquet") if not f.name.startswith("carry"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = pd.concat([pd.read_parquet(f, columns=["vol30"]) for f in files[::6]], ignore_index=True)["vol30"]
    p50, p90 = sample.quantile(0.5), sample.quantile(0.9)
    filters = {"all stocks": lambda d: d["vol30"].notna(), "volatile half": lambda d: d["vol30"] >= p50, "most volatile tenth": lambda d: d["vol30"] >= p90}
    rs = RuleScan(SCORES, filters)
    t0 = time.time()
    for i, f in enumerate(files):
        rs.add(pd.read_parquet(f))
        if i % 24 == 0:
            print(f.stem, f"{time.time() - t0:.0f} s", flush=True)
    r = rs.result()
    r.to_csv(out / "rule_check.csv", index=False, float_format="%.6g")
    n = len(r)
    null_max = sqrt(2 * log(n))
    L = [f"# Simple stock-picking rules against the cost of trading (2008-2026)", "",
         f"{n:,} tests (rule x decision minute x horizon). With that many tests the best |t| expected from pure luck is about {null_max:.1f}: only a t clearly above that, in most years, is evidence.",
         f"Volatility filter on the 30-minute realised volatility: volatile half >= {p50:.1f} bp a minute, most volatile tenth >= {p90:.1f}. 1 pick costs 0.08% (a whole 2,000-euro position), 3 picks 0.14% each (three slots), Tiered, 2 bp half-spread a side.", "",
         f"Rules with a positive result after costs: {int((r['net_bp'] > 0).sum())} of {n}; with net t above {null_max:.1f}: {int((r['net_t'] > null_max).sum())}; excess t above {null_max:.1f} or below -{null_max:.1f} (a real stock-picking effect, whatever the cost): {int((r['excess_t'].abs() > null_max).sum())}.", ""]
    hdr = "| Pick | Score | Filter | k | Minute | Hold | Gross | Excess over filter | Excess t | Net | Net t | Winning days | Years net > 0 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    row = lambda x: f"| {x['pick']} | {x['score']} | {x['filter']} | {x['k']} | {x['minute']} | {x['label'][3:]} | {x['gross_bp']:+.1f} bp | {x['excess_bp']:+.1f} bp | {x['excess_t']:+.1f} | {x['net_bp']:+.1f} bp | {x['net_t']:+.1f} | {x['win_net'] * 100:.0f}% | {x['net_years_positive']}/{x['years']} |"
    L += ["## Best by result after costs", "", hdr] + [row(x) for _, x in r.sort_values("net_t", ascending=False).head(20).iterrows()]
    L += ["", "## Strongest stock-picking effect (excess over the average filtered stock), before costs", "", hdr] + [row(x) for _, x in r.reindex(r["excess_t"].abs().sort_values(ascending=False).index).head(20).iterrows()]
    L += ["", "## Best net result per filter and number of picks", "", hdr]
    for (fl, k), g in r.groupby(["filter", "k"]):
        L.append(row(g.sort_values("net_bp", ascending=False).iloc[0]))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
