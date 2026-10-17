"""Minute features and forward returns from the Hugging Face minute bars (roadmap S14).

Each month's raw file is downloaded, reduced to the S&P 500 members of that month (plus SPY for the market
context), turned into one row per member, session and decision minute by `algotrader.minutefeat.process`, and
deleted. A month needs the previous month's carry file (last sessions' daily close and volume), so months run in
order; existing outputs are skipped, so an interrupted run resumes.

    python scripts/hf_minute_features.py 2008-01 2026-03 --out data/hf_minute --raw data/hf_raw --history data/history
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from hf_daily import URL, month_range                                          # noqa: E402

from algotrader.history import parse_members                                   # noqa: E402
from algotrader.minutefeat import MARKET, membership_by_day, month_tickers, process            # noqa: E402

COLS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


def read_month(path: Path, tickers: set) -> pd.DataFrame:
    """Rows of `tickers` in New York time: SPY with all hours, the others regular session only."""
    parts = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=2_000_000, columns=COLS):
        df = b.to_pandas()
        df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        t = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        minute = t.dt.hour * 60 + t.dt.minute
        keep = (df["ticker"] == MARKET) | ((minute >= 570) & (minute < 960))
        parts.append(df.assign(t=t)[keep].drop(columns="timestamp"))
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("first")
    p.add_argument("last")
    p.add_argument("--out", default="data/hf_minute")
    p.add_argument("--raw", default="data/hf_raw")
    p.add_argument("--history", default="data/history")
    args = p.parse_args(argv)
    out, raw = Path(args.out), Path(args.raw)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    members = parse_members((Path(args.history) / "sp500_members.csv").read_text())
    for m in month_range(args.first, args.last):
        dest = out / f"{m}.parquet"
        if dest.exists():
            continue
        t0 = time.time()
        start = pd.Period(m, "M").start_time
        end = pd.Period(m, "M").end_time
        tickers = month_tickers(members, start, end) | {MARKET}
        prev = out / f"carry_{(pd.Period(m, 'M') - 1).strftime('%Y-%m')}.parquet"
        carry = pd.read_parquet(prev) if prev.exists() else None
        if carry is None and m != args.first:
            print(m, "no carry file from the previous month: gap and daily features start empty", flush=True)
        src = raw / f"ohlcv_{m}.parquet"
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "--retry-delay", "10", "-o", str(src), URL.format(month=m)])
        if r.returncode != 0:
            print(m, "download failed", r.returncode, flush=True)
            src.unlink(missing_ok=True)
            continue
        df = read_month(src, tickers)
        src.unlink()
        days = pd.DatetimeIndex(sorted(df["t"].dt.normalize().unique()))
        mem = membership_by_day(members, days, sorted(set(df["ticker"])))
        rows, carry_out = process(df, mem, carry)
        carry_out.to_parquet(out / f"carry_{m}.parquet", index=False)
        rows.to_parquet(dest, index=False)
        print(m, f"{len(rows):,} rows, {rows['ticker'].nunique()} tickers, {rows['date'].nunique()} days, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
