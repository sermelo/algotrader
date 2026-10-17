"""Walk-forward gradient boosting on the minute features: does it beat the cost? (roadmap S14)

Pre-registered before any test year was seen (see `algotrader/minutemodel.py`): universe = the most volatile tenth of stocks at
each decision, one model per horizon (30, 90, 240 minutes, to the close), fitted every year on the earlier years only,
test years 2013-2026. The primary tests are 12: entries at 10:00, 10:30 and 11:00, holds of 90 and 240 minutes, one or
three picks; everything else is exploratory. Writes predictions, `summary.md` and result CSVs in `--out`.

    python scripts/minute_model.py --data data/hf_minute --out data/minute_model
"""
import argparse
import sys
import time
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.minutefeat import LABELS
from algotrader.minutemodel import HORIZONS, MODEL_FEATURES, WalkForward, best_of_day, information_coefficient, prepare, threshold_table
from algotrader.minutescan import RuleScan

COSTS = {1: 0.0008, 3: 0.0014}
PRIMARY = {"minute": (30, 60, 90), "label": ("fwd90", "fwd240"), "k": (1, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/hf_minute")
    p.add_argument("--out", default="data/minute_model")
    p.add_argument("--vol-rank", type=float, default=0.9)
    p.add_argument("--first-test-year", type=int, default=2013)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(a.data).glob("*.parquet") if not f.name.startswith("carry"))
    t0 = time.time()
    frames = {}
    for year in sorted({f.stem[:4] for f in files}):
        df = pd.concat([pd.read_parquet(f) for f in files if f.stem[:4] == year], ignore_index=True)
        frames[int(year)] = prepare(df, vol_rank=a.vol_rank).reset_index(drop=True)
        print(year, f"{len(frames[int(year)]):,} rows in the universe, {time.time() - t0:.0f} s", flush=True)
        del df
    wf = WalkForward(first_test_year=a.first_test_year)
    preds = wf.run(frames)
    print(f"fitted {len(wf.log)} models, {time.time() - t0:.0f} s", flush=True)
    preds.to_parquet(out / "predictions.parquet", index=False)

    L = ["# Walk-forward gradient boosting on the volatile tenth of S&P 500 stocks", "",
         f"Universe: top {1 - a.vol_rank:.0%} of the cross-section by 30-minute volatility at each decision; {len(preds):,} out-of-sample decisions, "
         f"{preds['date'].dt.year.min()}-{preds['date'].dt.year.max()}; features: {len(MODEL_FEATURES)}; one model per horizon, refitted every year on all earlier years.", "",
         "## Rank correlation between prediction and realised return (IC), by year", "",
         "| Year | " + " | ".join(h[3:] for h in HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
    ic = {h: information_coefficient(preds, h) for h in HORIZONS}
    for y in sorted(ic[HORIZONS[0]].index):
        L.append(f"| {y} | " + " | ".join(f"{ic[h].get(y, np.nan):+.4f}" for h in HORIZONS) + " |")
    L.append("| mean | " + " | ".join(f"{ic[h].mean():+.4f}" for h in HORIZONS) + " |")

    rs = RuleScan([f"pred_{h}" for h in HORIZONS], {"model universe": lambda d: d["date"].notna()}, ks=(1, 3), costs=COSTS)
    rs.add(preds.sort_values(["date", "minute", "ticker"]))
    r = rs.result()
    r = r[(r["pick"] == "high") & (r["score"].str[5:] == r["label"])]
    r.to_csv(out / "model_rules.csv", index=False, float_format="%.6g")
    n_tests = len(r)
    null_max = sqrt(2 * log(n_tests))
    prim = r[r["minute"].isin(PRIMARY["minute"]) & r["label"].isin(PRIMARY["label"]) & r["k"].isin(PRIMARY["k"])]
    hdr = "| Hold | Decision | Picks | Gross per trade | Excess over the volatile tenth | Net of cost | Net t | Winning trades | Years net > 0 |\n|---|---|---|---|---|---|---|---|---|"
    row = lambda x: f"| {x['label'][3:]} | {clock(x['minute'])} | {x['k']} | {x['gross_bp']:+.1f} bp | {x['excess_bp']:+.1f} bp (t {x['excess_t']:+.1f}) | {x['net_bp']:+.1f} bp | {x['net_t']:+.1f} | {x['win_net'] * 100:.0f}% | {x['net_years_positive']}/{x['years']} |"
    L += ["", f"## The 12 pre-registered tests (costs: 1 pick 0.08%, 3 picks 0.14% each)", "", hdr] + [row(x) for _, x in prim.sort_values(["label", "k", "minute"]).iterrows()]
    ok = prim[(prim["net_t"] > 3) & (prim["net_years_positive"] > prim["years"] / 2)]
    L += ["", f"Primary tests with net t above 3 and a positive result in most years: **{len(ok)} of {len(prim)}**.", "",
          f"## All {n_tests} (horizon x decision minute x picks) tests: the best", "", f"Luck alone would give a best t of about {null_max:.1f}.", "", hdr] + \
         [row(x) for _, x in r.sort_values("net_t", ascending=False).head(10).iterrows()]

    L += ["", "## The most confident trade of each day (one trade a day, any decision minute), by prediction threshold", ""]
    for h in ("fwd90", "fwd240", "fwd_close"):
        t = best_of_day(preds, h, COSTS[1])
        t.to_csv(out / f"best_of_day_{h}.csv", index=False, float_format="%.6g")
        L += [f"### Hold {h[3:]}: cost 0.08%", "", "| Predicted edge at least | Trades a year | Gross | Net | Net t | Winning trades |", "|---|---|---|---|---|---|"] + \
             [f"| {x.tau_bp:.0f} bp | {x.trades_per_year:.0f} | {x.gross_bp:+.1f} bp | {x.net_bp:+.1f} bp | {x.net_t:+.1f} | {x.win * 100:.0f}% |" for x in t.itertuples()] + [""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def clock(m):
    t = 570 + int(m)
    return f"{t // 60:02d}:{t % 60:02d}"


if __name__ == "__main__":
    sys.exit(main())
