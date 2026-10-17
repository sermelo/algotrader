"""The live-tradable version of the best-of-day result: first decision over a threshold (roadmap S14).

`best_of_day` chose each day's best prediction in hindsight. Here a trader goes through the day in time order and takes the best
stock at the first decision whose predicted edge reaches `tau`, one trade a day. Walk-forward thresholds: for each test year the
threshold is the one with the best net result in the previous test years (from 2014), never the year itself. Writes `online.md`.

    python scripts/minute_model_online.py --model data/minute_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.minutemodel import first_passage

TAUS = (0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002)
COST = 0.0008


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="data/minute_model")
    a = p.parse_args(argv)
    preds = pd.read_parquet(Path(a.model) / "predictions.parquet")
    L = ["# Live-tradable rule: the first decision of the day where the best stock's predicted edge reaches a threshold", "",
         f"One trade a day, cost {COST * 1e4:.0f} bp, out-of-sample 2013-2026 ({preds['date'].nunique():,} days).", ""]
    for label in ("fwd90", "fwd240", "fwd_close"):
        t = first_passage(preds, label, COST, TAUS)
        L += [f"## Hold {label[3:]}: all years, by threshold", "", "| Threshold | Trades a year | Average decision (minutes after 09:30) | Gross | Net | Net t | Winning trades |", "|---|---|---|---|---|---|---|"]
        L += [f"| {x.tau_bp:.0f} bp | {x.trades_per_year:.0f} | {x.mean_minute:.0f} | {x.gross_bp:+.1f} bp | {x.net_bp:+.1f} bp | {x.net_t:+.1f} | {x.win * 100:.0f}% |" for x in t.itertuples()] + [""]
        # walk-forward threshold choice: best net-per-year (net bp x trades a year) over the earlier test years
        years = sorted(preds["date"].dt.year.unique())
        by_year = {y: first_passage(preds[preds["date"].dt.year == y], label, COST, TAUS).set_index("tau_bp") for y in years}
        chosen, picks = [], []
        for y in years[1:]:
            past = sum(by_year[k]["net_bp"].fillna(0) * by_year[k]["trades"] for k in years if k < y)
            tau = past.idxmax()
            row = by_year[y].loc[tau]
            chosen.append((y, tau, row["trades"], row["net_bp"]))
            sub = preds[preds["date"].dt.year == y]
            day = first_passage(sub, label, COST, (tau / 1e4,))
        tot_trades = sum(c[2] for c in chosen)
        tot_net = sum(c[2] * c[3] for c in chosen if np.isfinite(c[3])) / max(tot_trades, 1)
        L += [f"### Hold {label[3:]}: threshold chosen on earlier years only", "", "| Year | Threshold chosen | Trades | Net per trade |", "|---|---|---|---|"]
        L += [f"| {y} | {tau:.0f} bp | {int(n)} | {net:+.1f} bp |" for y, tau, n, net in chosen]
        L += ["", f"All {len(chosen)} years together: {int(tot_trades)} trades, {tot_net:+.1f} bp net per trade.", ""]
    (Path(a.model) / "online.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
