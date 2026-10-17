"""Is the model's "best trade of the day" result real? Audit of `scripts/minute_model.py` predictions (roadmap S14).

The most confident trade of each day looked far too good (+39 bp net, t about 11). Reasons it could be an artefact: bad prints in the
last-trade data (a false low print makes a "reversal" that no one could trade), bid-ask bounce in thinly traded names
(the print was the bid, the buyer pays the ask), a handful of huge winners, a very wide real spread in volatile names.
This joins the predictions with the features, describes the picked trades, repeats the best-of-day test with liquidity and sanity
filters applied BEFORE the pick, stresses the cost, and shows the biggest trades. Writes `audit.md` in `--out`.

    python scripts/minute_model_audit.py --data data/hf_minute --model data/minute_model
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.minutemodel import best_of_day

KEYS = ["date", "minute", "ticker"]
COLS = KEYS + ["price", "active30", "vol30", "logdv20", "range_bp", "rvol30", "ret5", "ret30", "gap", "ret_open"]
LABEL = "fwd90"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/hf_minute")
    p.add_argument("--model", default="data/minute_model")
    a = p.parse_args(argv)
    preds = pd.read_parquet(Path(a.model) / "predictions.parquet")
    parts = []
    for year, g in preds.groupby(preds["date"].dt.year):
        f = pd.concat([pd.read_parquet(x, columns=COLS) for x in sorted(Path(a.data).glob(f"{year}-*.parquet")) if not x.name.startswith("carry")], ignore_index=True)
        parts.append(g.merge(f, on=KEYS, how="left"))
        del f
    d = pd.concat(parts, ignore_index=True)
    d["abs_ret5"] = d["ret5"].abs()
    cost = 0.0008
    L = ["# Audit of the model's best trade of the day", "", f"{len(d):,} out-of-sample decisions in the volatile tenth, {d['date'].nunique():,} days.", ""]

    def picks(frame, label):
        p = frame[KEYS + [f"pred_{label}", label] + [c for c in COLS if c not in KEYS] + ["abs_ret5"]].dropna(subset=[f"pred_{label}", label])
        return p.loc[p.groupby("date")[f"pred_{label}"].idxmax()]

    for label in ("fwd90", "fwd240"):
        b = picks(d, label)
        g = b[label]
        L += [f"## Hold {label[3:]} minutes: the {len(b):,} best-of-day trades (no filter)", "",
              f"Mean gross {g.mean() * 1e4:+.1f} bp, median {g.median() * 1e4:+.1f} bp, mean after cutting the biggest and smallest 1%: {g.clip(g.quantile(.01), g.quantile(.99)).mean() * 1e4:+.1f} bp, after cutting 5%: {g.clip(g.quantile(.05), g.quantile(.95)).mean() * 1e4:+.1f} bp. "
              f"Trades gaining over 5%: {(g > 0.05).mean() * 100:.1f}%, losing over 5%: {(g < -0.05).mean() * 100:.1f}%; best {g.max() * 100:+.1f}%, worst {g.min() * 100:+.1f}%.", ""]
        uni = d[["price", "active30", "vol30", "logdv20", "range_bp", "rvol30", "abs_ret5"]].median()
        pk = b[["price", "active30", "vol30", "logdv20", "range_bp", "rvol30", "abs_ret5"]].median()
        L += ["| | Price | Trades in last 30 min | 30-min volatility (bp/min) | log dollar volume | Bar range (bp) | Volume surge | Last 5-min move |", "|---|---|---|---|---|---|---|---|",
              "| Picked trades (median) | " + " | ".join(f"{pk[c]:.3g}" for c in pk.index) + " |", "| Volatile tenth (median) | " + " | ".join(f"{uni[c]:.3g}" for c in uni.index) + " |", ""]
        top = b.sort_values(label, ascending=False).head(10)
        L += ["Ten biggest winners:", "", "| Date | Time | Ticker | Price | Last 5-min move | Opening gap | Volume surge | Predicted | Realised |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            t = 570 + int(r["minute"])
            L.append(f"| {r['date'].date()} | {t // 60:02d}:{t % 60:02d} | {r['ticker']} | {r['price']:.2f} | {r['ret5'] * 100:+.1f}% | {r['gap'] * 100:+.1f}% | {r['rvol30']:.1f}x | {r[f'pred_{label}'] * 1e4:+.0f} bp | {r[label] * 100:+.1f}% |")
        L.append("")

    filters = {"no filter": lambda x: x["price"] > 0,
               "traded in 25+ of the last 30 minutes": lambda x: x["active30"] >= 25,
               "price at least $50": lambda x: x["price"] >= 50,
               "last 5-min move under 3%": lambda x: x["abs_ret5"] <= 0.03,
               "opening gap under 5%": lambda x: x["gap"].abs() <= 0.05,
               "liquid (top half by dollar volume) and traded 25+ of 30 minutes": lambda x: (x["logdv20"] >= d["logdv20"].median()) & (x["active30"] >= 25),
               "all of: liquid, 25+ of 30 minutes, price $50+, move under 3%, gap under 5%": lambda x: (x["logdv20"] >= d["logdv20"].median()) & (x["active30"] >= 25) & (x["price"] >= 50) & (x["abs_ret5"] <= 0.03) & (x["gap"].abs() <= 0.05)}
    L += ["## Best-of-day with the filter applied before the pick (cost 0.08%)", ""]
    for label in ("fwd90", "fwd240"):
        L += [f"### Hold {label[3:]} minutes", "", "| Filter | Candidates kept | Trades a year | Gross | Median gross | Net | Net t | Winning trades |", "|---|---|---|---|---|---|---|---|"]
        for name, fn in filters.items():
            sub = d[fn(d)]
            t = best_of_day(sub, label, cost, taus=(0.0,)).iloc[0]
            med = picks(sub, label)[label].median() * 1e4
            L.append(f"| {name} | {len(sub) / len(d) * 100:.0f}% | {t.trades_per_year:.0f} | {t.gross_bp:+.1f} bp | {med:+.1f} bp | {t.net_bp:+.1f} bp | {t.net_t:+.1f} | {t.win * 100:.0f}% |")
        L.append("")
    L += ["## Cost stress: net per trade after extra round-trip costs (hold 90 minutes)", "", "| Extra cost | No filter | All filters |", "|---|---|---|"]
    allf = d[filters[list(filters)[-1]](d)]
    for extra in (0, 5, 10, 20, 40):
        L.append(f"| +{extra} bp | " + " | ".join(f"{best_of_day(s, 'fwd90', cost + extra / 1e4, taus=(0.0,)).iloc[0].net_bp:+.1f} bp (t {best_of_day(s, 'fwd90', cost + extra / 1e4, taus=(0.0,)).iloc[0].net_t:+.1f})" for s in (d, allf)) + " |")
    b = picks(d, "fwd90")
    # market stress of the day: median 30-minute volatility of the volatile tenth across all of the day's decisions
    stress = d.groupby("date")["vol30"].median()
    b = b.assign(stress=b["date"].map(stress))
    b["stress_q"] = pd.qcut(b["stress"], 4, labels=["calm", "normal", "busy", "stressed"])
    L += ["", "## By market stress (hold 90, no filter)", "", "Stress = the day's median 30-minute volatility in the volatile tenth, in quarters.", "", "| Stress | Trades | Gross | Median gross | Net | Net t | Winning trades |", "|---|---|---|---|---|---|---|"]
    for q, g in b.groupby("stress_q", observed=True):
        n = g["fwd90"] - cost
        L.append(f"| {q} | {len(g)} | {g['fwd90'].mean() * 1e4:+.1f} bp | {g['fwd90'].median() * 1e4:+.1f} bp | {n.mean() * 1e4:+.1f} bp | {n.mean() / (n.std() / np.sqrt(len(n))):+.1f} | {(n > 0).mean() * 100:.0f}% |")
    b["pred_q"] = pd.qcut(b["pred_fwd90"], 5, labels=False, duplicates="drop")
    L += ["", "## Is the predicted edge calibrated? (hold 90, best trade of each day, by fifths of the prediction)", "", "| Fifth | Predicted edge | Realised gross | Realised net | Net t |", "|---|---|---|---|---|"]
    for q, g in b.groupby("pred_q"):
        n = g["fwd90"] - cost
        L.append(f"| {int(q) + 1} | {g['pred_fwd90'].mean() * 1e4:+.1f} bp | {g['fwd90'].mean() * 1e4:+.1f} bp | {n.mean() * 1e4:+.1f} bp | {n.mean() / (n.std() / np.sqrt(len(n))):+.1f} |")
    b["clock"] = ((570 + b["minute"]) // 60).astype(int).astype(str) + ":" + (((570 + b["minute"]) % 60).astype(int)).astype(str).str.zfill(2)
    L += ["", "## When the best trade happens (hold 90)", "", "| Decision at | Share of days | Net per trade |", "|---|---|---|"]
    for c, g in b.groupby("clock"):
        L.append(f"| {c} | {len(g) / len(b) * 100:.0f}% | {(g['fwd90'] - cost).mean() * 1e4:+.1f} bp |")
    b.to_parquet(Path(a.model) / "best_of_day_trades.parquet", index=False)
    net = (b["fwd90"] - cost).groupby(b["date"].dt.year).mean() * 1e4
    L += ["", "## By year (hold 90, no filter): net per trade", "", "| Year | " + " | ".join(str(y) for y in net.index) + " |", "|---|" + "---|" * len(net), "| net bp | " + " | ".join(f"{v:+.0f}" for v in net.values) + " |"]
    (Path(a.model) / "audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
