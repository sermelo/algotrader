"""Could the best-of-day trades really have been filled? Replay on the raw minute bars (roadmap S14).

Takes the trades of `minute_model_audit.py` (`best_of_day_trades.parquet`), downloads every `--every`-th month of the Hugging Face
minute bars, and re-prices each trade of those months under harsher fills (`algotrader.minutemodel.replay_fills`): buying at the
entry minute's high and selling at the exit minute's low, entering 1, 2 and 5 minutes late, and a quarter of each bar's range as
slippage. The plain fill must reproduce the dataset's label. Writes `replay.md` and `replay.csv`.

    python scripts/minute_trade_replay.py --model data/minute_model --every 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hf_daily import URL                                                      # noqa: E402
from hf_minute_features import read_month                                      # noqa: E402

from algotrader.minutemodel import replay_fills                                # noqa: E402

COST = 0.0008


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="data/minute_model")
    p.add_argument("--raw", default="data/hf_raw")
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--hold", type=int, default=90)
    a = p.parse_args(argv)
    trades = pd.read_parquet(Path(a.model) / "best_of_day_trades.parquet")
    trades["month"] = trades["date"].dt.strftime("%Y-%m")
    months = sorted(trades["month"].unique())[::a.every]
    raw = Path(a.raw)
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in months:
        sub = trades[trades["month"] == m]
        src = raw / f"ohlcv_{m}.parquet"
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "--retry-delay", "10", "-o", str(src), URL.format(month=m)])
        if r.returncode != 0:
            print(m, "download failed", flush=True)
            src.unlink(missing_ok=True)
            continue
        df = read_month(src, set(sub["ticker"]))
        src.unlink()
        df = df.assign(day=df["t"].dt.normalize(), m=df["t"].dt.hour * 60 + df["t"].dt.minute - 570)
        df = df[(df["m"] >= 0) & (df["m"] < 390)]
        for _, t in sub.iterrows():
            b = df[(df["ticker"] == t["ticker"]) & (df["day"] == t["date"])].drop_duplicates("m", keep="last").set_index("m").sort_index()
            res = replay_fills(b[["open", "high", "low", "close"]], int(t["minute"]), a.hold) if len(b) else None
            if res is not None:
                rows.append({"date": t["date"], "ticker": t["ticker"], "minute": t["minute"], "label": t["fwd90"], **res})
        print(m, len(rows), "trades replayed", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(Path(a.model) / "replay.csv", index=False, float_format="%.6g")
    L = ["# Replay of the best-of-day trades on the raw minute bars", "",
         f"{len(r):,} of {len(trades[trades['month'].isin(months)]):,} trades in {len(months)} sampled months (every {a.every}rd month) could be replayed (a trade needs a print in its entry minute). Hold {a.hold} minutes, cost {COST * 1e4:.0f} bp.", "",
         f"Check: the plain fill reproduces the dataset's label on {np.mean(np.isclose(r['base'], r['label'], atol=1e-5)) * 100:.1f}% of trades (mean difference {(r['base'] - r['label']).abs().mean() * 1e4:.2f} bp).", "",
         "| Fill | Gross per trade | Net of 8 bp | Net t | Median gross | Winning trades after cost |", "|---|---|---|---|---|---|"]
    names = {"base": "plain: buy the entry minute's open, sell the exit minute's open", "delay1": "entry 1 minute late", "delay2": "entry 2 minutes late", "delay5": "entry 5 minutes late",
             "range_slip": "slippage: a quarter of each bar's range against us", "worst_bar": "worst case: buy the entry minute's high, sell the exit minute's low"}
    for k, label in names.items():
        x = r[k].dropna()
        n = x - COST
        L.append(f"| {label} | {x.mean() * 1e4:+.1f} bp | {n.mean() * 1e4:+.1f} bp | {n.mean() / (n.std() / np.sqrt(len(n))):+.1f} | {x.median() * 1e4:+.1f} bp | {(n > 0).mean() * 100:.0f}% |")
    (Path(a.model) / "replay.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
