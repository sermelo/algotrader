"""Placebo and ablation tests of the model's best-of-day result (roadmap S14).

The best trade of each day looked far too good (see `minute_model_audit.py`). If it comes from the pipeline and not from information, it
survives where it should not: random picks, a model trained on shuffled targets, a model that only knows which stock it is looking at.
Each variant is walk-forward like the real model (hold 90 minutes) and scored as best-of-day, cost 0.08%.

    python scripts/minute_model_placebo.py --data data/hf_minute --out data/minute_model
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.minutemodel import MODEL_FEATURES, WalkForward, best_of_day, information_coefficient, prepare

LABEL = "fwd90"
IDENTITY = ["price", "logdv20", "minute"]
EVENT = ["gap", "z_gap", "rvol30", "rvol_day", "ret_open", "z_open", "minute", "mins_left"]
VARIANTS = {"real model (all features)": dict(features=MODEL_FEATURES),
            "shuffled targets (placebo)": dict(features=MODEL_FEATURES, shuffle_seed=1),
            "identity features only (price, dollar volume, minute)": dict(features=IDENTITY),
            "without price and dollar volume": dict(features=[f for f in MODEL_FEATURES if f not in ("price", "logdv20")]),
            "event features only (gap, volume surge, move since open, minute)": dict(features=EVENT),
            "short-term features only (returns, VWAP, range, SPY)": dict(features=[f for f in MODEL_FEATURES if f in ("ret5", "ret15", "ret30", "ret60", "z5", "z30", "vwap_dist", "range_pos", "rel30", "spy_ret30", "vol30", "range_bp", "active30", "minute")])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/hf_minute")
    p.add_argument("--out", default="data/minute_model")
    a = p.parse_args(argv)
    files = sorted(f for f in Path(a.data).glob("*.parquet") if not f.name.startswith("carry"))
    frames = {}
    for year in sorted({f.stem[:4] for f in files}):
        df = pd.concat([pd.read_parquet(f) for f in files if f.stem[:4] == year], ignore_index=True)
        frames[int(year)] = prepare(df, horizons=(LABEL,)).reset_index(drop=True)
        del df
    rows, t0 = [], time.time()
    ref = None
    for name, kw in VARIANTS.items():
        wf = WalkForward(first_test_year=2013, horizons=(LABEL,), **kw)
        pr = wf.run(frames)
        ref = pr if ref is None else ref
        ic = information_coefficient(pr, LABEL).mean()
        t = best_of_day(pr, LABEL, 0.0008, taus=(0.0,)).iloc[0]
        rows.append((name, ic, t))
        print(name, f"{time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(0)
    rnd = ref.assign(pred_fwd90=rng.normal(size=len(ref)))
    t = best_of_day(rnd, LABEL, 0.0008, taus=(0.0,)).iloc[0]
    rows.append(("random pick among the volatile tenth", 0.0, t))
    L = ["# Placebo and ablation tests of the best-of-day result (hold 90 minutes, cost 0.08%)", "",
         "| Variant | Mean IC | Trades a year | Gross per trade | Net per trade | Net t | Winning trades |", "|---|---|---|---|---|---|---|"]
    for name, ic, t in rows:
        L.append(f"| {name} | {ic:+.4f} | {t.trades_per_year:.0f} | {t.gross_bp:+.1f} bp | {t.net_bp:+.1f} bp | {t.net_t:+.1f} | {t.win * 100:.0f}% |")
    (Path(a.out) / "placebo.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
