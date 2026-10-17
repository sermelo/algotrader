"""Distribution of price moves over 1 minute to 4 hours against the cost of a trade (roadmap S14).

Downloads a sample of months of the Hugging Face minute bars (October of each year by default), keeps the
S&P 500 members costing at least `--min-price`, measures the change of the last trade price over each horizon for
every start minute from 10:00 to 15:00 (`algotrader.movestats`) and writes, in `--out`: `move_stats.json` (all numbers
and histograms), `summary.md` (tables) and `move_distributions.svg` (histograms with the normal distribution of the
same spread, and the break-even thresholds).

    python scripts/minute_move_stats.py --out data/move_stats --raw data/hf_raw --history data/history
"""
import argparse
import json
import subprocess
import sys
import time
from math import exp, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hf_daily import URL                                                      # noqa: E402
from hf_minute_features import read_month                                      # noqa: E402

from algotrader.history import parse_members                                   # noqa: E402
from algotrader.minutefeat import MARKET, membership_by_day, month_tickers     # noqa: E402
from algotrader.movestats import EDGES, HORIZONS, THRESHOLDS, Accumulator, window_changes   # noqa: E402

PANELS = (1, 5, 10, 30, 90, 240)


def svg(stats: dict, hists: dict, thresholds, path: Path):
    """Histograms with the normal curve of the same spread; the dashed lines mark the middle break-even."""
    W, H, cols = 1200, 740, 3
    pw, ph = W / cols, H / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="monospace" font-size="12"><rect width="{W}" height="{H}" fill="#ffffff"/>']
    c = thresholds[len(thresholds) // 2]
    for i, h in enumerate(PANELS):
        s = stats[str(h)]
        sd, hist = s["sd"], np.array(hists[str(h)], dtype=float)
        x0, y0 = (i % cols) * pw, (i // cols) * ph
        lim = max(4 * sd, 1.6 * c)
        centers = (EDGES[:-1] + EDGES[1:]) / 2
        sel = np.abs(centers) <= lim
        step = max(1, int(round(lim * 2 / 0.0001 / 70)))                    # about 70 bars across the panel
        xs, ys = centers[sel], hist[sel]
        n = len(xs) // step * step
        xs, ys = xs[:n].reshape(-1, step).mean(1), ys[:n].reshape(-1, step).sum(1)
        dens = ys / ys.sum() / (step * 0.0001)                              # share per unit change
        normal = lambda x: exp(-x * x / (2 * sd * sd)) / (sd * sqrt(2 * pi))
        top = dens.max() * 1.15
        L, R, T, B = x0 + 16, x0 + pw - 16, y0 + 38, y0 + ph - 34
        X = lambda v: L + (v + lim) / (2 * lim) * (R - L)
        Y = lambda d: B - d / top * (B - T)
        out.append(f'<text x="{L}" y="{y0 + 18}" font-weight="bold">{h} minute{"s" if h > 1 else ""}: spread {sd * 100:.2f}%</text>')
        out.append(f'<text x="{L}" y="{y0 + 32}" fill="#555">beyond ±{c * 100:.2f}%: {s["tails"][len(thresholds) // 2]["either"] * 100:.0f}% of windows '
                   f'(normal {s["tails"][len(thresholds) // 2]["normal_either"] * 100:.0f}%)</text>')
        bw = (R - L) / len(xs)
        for x, d in zip(xs, dens):
            col = "#c23b3a" if abs(x) > c else "#2a78d6"
            out.append(f'<rect x="{X(x) - bw / 2 + 0.5:.1f}" y="{Y(d):.1f}" width="{max(bw - 1, 0.5):.1f}" height="{B - Y(d):.1f}" fill="{col}" fill-opacity="0.55"/>')
        pts = " ".join(f"{X(v):.1f},{Y(normal(v)):.1f}" for v in np.linspace(-lim, lim, 120))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#14181c" stroke-width="1.5"/>')
        rs = s["robust_sd"]
        pts = " ".join(f"{X(v):.1f},{Y(exp(-v * v / (2 * rs * rs)) / (rs * sqrt(2 * pi))):.1f}" for v in np.linspace(-lim, lim, 240))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#1a7f4b" stroke-width="1.5" stroke-dasharray="5 3"/>')
        for sgn in (-1, 1):
            out.append(f'<line x1="{X(sgn * c):.1f}" x2="{X(sgn * c):.1f}" y1="{T}" y2="{B}" stroke="#14181c" stroke-dasharray="4 3" stroke-width="1"/>')
        out.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="#737b82"/>')
        for v in (-lim, -lim / 2, 0, lim / 2, lim):
            anchor = "start" if v == -lim else "end" if v == lim else "middle"
            out.append(f'<text x="{X(v):.1f}" y="{B + 14}" text-anchor="{anchor}" fill="#555">{v * 100:+.1f}%</text>')
    out.append(f'<text x="16" y="{H - 20}" fill="#555">Bars: windows by price change (red: beyond the break-even ±{c * 100:.2f}%, dashed vertical).</text>')
    out.append(f'<text x="16" y="{H - 6}" fill="#555">Black line: normal with the same standard deviation; green dashed: normal with the same interquartile range.</text>')
    out.append("</svg>")
    path.write_text("\n".join(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--months", nargs="*", default=[f"{y}-10" for y in range(2008, 2026)])
    p.add_argument("--out", default="data/move_stats")
    p.add_argument("--raw", default="data/hf_raw")
    p.add_argument("--history", default="data/history")
    p.add_argument("--min-price", type=float, default=20.0)
    p.add_argument("--thresholds", nargs="*", type=float, default=list(THRESHOLDS))
    a = p.parse_args(argv)
    out, raw = Path(a.out), Path(a.raw)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    thr = tuple(a.thresholds)
    members = parse_members((Path(a.history) / "sp500_members.csv").read_text())
    total = {h: Accumulator(thr) for h in HORIZONS}
    years = {}
    for m in a.months:
        t0 = time.time()
        per = pd.Period(m, "M")
        tickers = month_tickers(members, per.start_time, per.end_time) | {MARKET}
        src = raw / f"ohlcv_{m}.parquet"
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "--retry-delay", "10", "-o", str(src), URL.format(month=m)])
        if r.returncode != 0:
            print(m, "download failed", flush=True)
            src.unlink(missing_ok=True)
            continue
        df = read_month(src, tickers)
        src.unlink()
        days = pd.DatetimeIndex(sorted(df["t"].dt.normalize().unique()))
        mem = membership_by_day(members, days, sorted(set(df["ticker"])))
        y = years.setdefault(m[:4], {h: Accumulator(thr) for h in HORIZONS})
        for h, x in window_changes(df, mem, HORIZONS, min_price=a.min_price):
            total[h].add(x)
            y[h].add(x)
        print(m, f"{int(total[HORIZONS[0]].n):,} windows so far, {time.time() - t0:.0f} s", flush=True)
    stats = {str(h): total[h].summary() for h in HORIZONS}
    res = {"months": a.months, "min_price": a.min_price, "thresholds": thr, "stats": stats,
           "hist": {str(h): total[h].hist.tolist() for h in HORIZONS}, "edges": [float(EDGES[0]), float(EDGES[-1]), len(EDGES) - 1],
           "by_year": {y: {str(h): acc[h].summary() for h in HORIZONS} for y, acc in years.items()}}
    (out / "move_stats.json").write_text(json.dumps(res))
    svg(stats, res["hist"], thr, out / "move_distributions.svg")
    lines = [f"Price change of the last trade over h minutes, S&P 500 members costing at least ${a.min_price:.0f}, "
             f"start minutes 10:00-15:00, months: {', '.join(a.months)}.", "",
             "| Horizon | Windows | Mean | Spread (sd) | Robust spread | Mean |move| | Zero moves | Excess kurtosis | " +
             " | ".join(f"Up > {c * 100:.2f}% | Down > {c * 100:.2f}% | Either (normal)" for c in thr) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|---|---|" * len(thr)]
    for h in HORIZONS:
        s = stats[str(h)]
        lines.append(f"| {h} min | {s['n']:,} | {s['mean'] * 100:+.4f}% | {s['sd'] * 100:.3f}% | {s['robust_sd'] * 100:.3f}% | {s['mean_abs'] * 100:.3f}% | "
                     f"{s['zero_share'] * 100:.0f}% | {s['excess_kurtosis']:.1f} | " +
                     " | ".join(f"{t['up'] * 100:.1f}% | {t['down'] * 100:.1f}% | {t['either'] * 100:.1f}% ({t['normal_either'] * 100:.1f}%)" for t in s["tails"]) + " |")
    mid = len(thr) // 2
    lines += ["", f"Share of windows moving more than {thr[mid] * 100:.2f}% either way, by year:", "",
              "| Year | " + " | ".join(f"{h} min" for h in HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
    for y in sorted(years):
        lines.append(f"| {y} | " + " | ".join(f"{res['by_year'][y][str(h)]['tails'][mid]['either'] * 100:.0f}%" for h in HORIZONS) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
