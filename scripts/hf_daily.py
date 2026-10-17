"""Aggregate the Hugging Face OHLCV-1m monthly files into regular-session daily bars.

Source: https://huggingface.co/datasets/mito0o852/OHLCV-1m (US minute bars 1992 to 2026,
UTC timestamps, NOT adjusted for splits or dividends, extended hours included, licence
not stated). Each month is downloaded, reduced to one row per ticker and New York trading
day (09:30 <= time < 16:00: first open, max high, min low, last close, summed volume, bar
count, plus the open of the 16:00 and 13:00 bars, where the closing-auction print sits)
and the raw file is deleted, so the 85 GB never sit on disk together.

    python scripts/hf_daily.py 2008-01 2018-03 --out data/hf_daily [--raw data/hf_raw]

Raw files are untrusted data: they are only read with pyarrow. Existing outputs are
skipped, so an interrupted run resumes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data/ohlcv_{month}.parquet"
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60
EARLY_CLOSE_MIN = 13 * 60


def month_range(first: str, last: str) -> list:
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def early_close_dates(path: Path) -> frozenset:
    """Sessions that closed at 13:00: almost no bars between 13:05 and 16:00 compared with
    the morning (after-hours prints only). Normal days have about as many bars after 13:05
    as before 13:00."""
    morning, late = {}, {}
    for b in pq.ParquetFile(path).iter_batches(batch_size=4_000_000, columns=["timestamp"]):
        ts = b.to_pandas()["timestamp"].dt.tz_convert("America/New_York")
        minute, date = ts.dt.hour * 60 + ts.dt.minute, ts.dt.tz_localize(None).dt.normalize()
        for store, sel in ((morning, (minute >= OPEN_MIN) & (minute < EARLY_CLOSE_MIN)),
                           (late, (minute > EARLY_CLOSE_MIN + 4) & (minute < CLOSE_MIN))):
            for d, n in date[sel].value_counts().items():
                store[d] = store.get(d, 0) + n
    return frozenset(d for d, m in morning.items() if m > 0 and late.get(d, 0) < 0.2 * m)


def reduce_batch(df: pd.DataFrame, early: frozenset = frozenset()) -> pd.DataFrame:
    """Per (ticker, session date) partial aggregates of one batch of minute bars. Sessions
    in `early` end at 13:00."""
    ts = df["timestamp"].dt.tz_convert("America/New_York")
    minute = ts.dt.hour * 60 + ts.dt.minute
    df = df.assign(date=ts.dt.tz_localize(None).dt.normalize(), t=ts.dt.tz_localize(None),
                   minute=minute).sort_values("t")
    cross = {}
    for name, m in (("x1600", CLOSE_MIN), ("x1300", EARLY_CLOSE_MIN)):
        c = df[df["minute"] == m]
        cross[name] = c.groupby(["ticker", "date"], sort=False)["open"].first()
    end = pd.Series(CLOSE_MIN, index=df.index).where(~df["date"].isin(early), EARLY_CLOSE_MIN)
    df = df[(df["minute"] >= OPEN_MIN) & (df["minute"] < end)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["ticker", "date"], sort=False)
    out = pd.DataFrame({
        "t_first": g["t"].first(), "open": g["open"].first(),
        "t_last": g["t"].last(), "close": g["close"].last(),
        "high": g["high"].max(), "low": g["low"].min(),
        "volume": g["volume"].sum(), "nbars": g["close"].size()})
    for name, s in cross.items():
        out[name] = s.reindex(out.index)
    return out.reset_index()


def combine(parts: list) -> pd.DataFrame:
    """Merge partial aggregates (a ticker-day can span batches) into daily bars."""
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    first = df.sort_values("t_first").groupby(["ticker", "date"], sort=False)[["open"]].first()
    last = df.sort_values("t_last").groupby(["ticker", "date"], sort=False)[["close"]].last()
    g = df.groupby(["ticker", "date"], sort=False)
    out = pd.DataFrame({"open": first["open"], "high": g["high"].max(), "low": g["low"].min(),
                        "close": last["close"], "volume": g["volume"].sum(), "nbars": g["nbars"].sum()})
    for name in ("x1600", "x1300"):
        out[name] = g[name].first()                  # first non-null: the bar is in one batch only
    return out.reset_index()


def aggregate_file(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    early = early_close_dates(path)
    parts = [reduce_batch(b.to_pandas(), early) for b in pf.iter_batches(batch_size=2_000_000)]
    return combine(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("first")
    p.add_argument("last")
    p.add_argument("--out", default="data/hf_daily")
    p.add_argument("--raw", default="data/hf_raw")
    args = p.parse_args(argv)
    out, raw = Path(args.out), Path(args.raw)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    for m in month_range(args.first, args.last):
        dest = out / f"{m}.parquet"
        if dest.exists():
            continue
        src = raw / f"ohlcv_{m}.parquet"
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "--retry-delay", "10", "-o", str(src),
                            URL.format(month=m)])
        if r.returncode != 0:
            print(m, "download failed", r.returncode, flush=True)
            src.unlink(missing_ok=True)
            continue
        daily = aggregate_file(src)
        src.unlink()
        daily.to_parquet(dest, index=False)
        print(m, len(daily), "ticker-days", daily["ticker"].nunique(), "tickers", flush=True)


if __name__ == "__main__":
    sys.exit(main())
