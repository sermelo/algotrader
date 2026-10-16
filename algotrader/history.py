"""Long daily price history for a point-in-time S&P 500 universe (roadmap item D1).

Sources (all free; see ROADMAP.md "D1 findings"):
- membership: daily S&P 500 member lists (GitHub hanshof/sp500_constituents, MIT);
- Quandl WIKI prices (Kaggle marketneutral/quandl-wiki-prices-us-equites): adjusted
  daily OHLCV of about 3,200 US stocks including delisted ones, up to 2018-03-27;
- Tiingo end-of-day API (free account): adjusted daily OHLCV including delisted stocks.

Each series is saved as `<out>/<source>/<TICKER>.parquet` with a date index and
adjusted `open, high, low, close, volume`. Tickers are renamed and reused over time,
so `check` compares every series with the membership period and with the other
sources where they overlap before it is used.

The Tiingo token is read from ~/.config/tiingo/token and never printed. The free plan
allows few requests (500 symbols a month, 50 requests an hour, 1,000 a day), so
`tiingo` is resumable: it logs every request and stops at the limits.

Example:
    python -m algotrader.history membership          # -> data/history/sp500_members.csv
    python -m algotrader.history wiki --zip data/kaggle/wiki/quandl-wiki-prices-us-equites.zip
    python -m algotrader.history tiingo --max 50      # fetch up to 50 more tickers
    python -m algotrader.history check                # -> data/history/coverage.csv
"""
import argparse
import csv
import datetime as dt
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERS_URL = ("https://raw.githubusercontent.com/hanshof/sp500_constituents/main/"
               "sp_500_historical_components.csv")
TIINGO_LIST_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
TIINGO_PRICES = "https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}&format=csv"
TIINGO_TOKEN = Path.home() / ".config" / "tiingo" / "token"
OUT = Path("data/history")
START = "2008-01-01"
COLUMNS = ["open", "high", "low", "close", "volume"]
UA = {"User-Agent": "algotrader/0.1"}


def _get(url, headers=None, opener=urllib.request.urlopen) -> bytes:
    with opener(urllib.request.Request(url, headers=UA | (headers or {}))) as resp:
        return resp.read()


# ---------------------------------------------------------------- membership

def parse_members(text: str) -> pd.DataFrame:
    """Long table (date, ticker) from the `date,tickers` CSV (tickers comma-joined)."""
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        for t in rec["tickers"].split(","):
            t = t.strip()
            if t:
                rows.append((rec["date"], t))
    df = pd.DataFrame(rows, columns=["date", "ticker"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def member_spans(members: pd.DataFrame, start=START) -> pd.DataFrame:
    """First and last membership date per ticker, for members on or after `start`."""
    m = members[members["date"] >= pd.Timestamp(start)]
    g = m.groupby("ticker")["date"]
    return pd.DataFrame({"first": g.min(), "last": g.max()}).sort_index()


def tiingo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def wiki_symbol(ticker: str) -> str:
    return ticker.replace(".", "_")


# ---------------------------------------------------------------- WIKI (Kaggle)

def wiki_extract(zip_path, out_dir, tickers=None, chunksize=500_000) -> list:
    """Write adjusted OHLCV per ticker from the WIKI_PRICES.csv inside `zip_path`.

    The file is sorted by ticker, so it is streamed in chunks and each ticker is written
    once complete. `tickers` (WIKI spelling) limits which are kept. Returns tickers written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(tickers) if tickers is not None else None
    cols = {"adj_open": "open", "adj_high": "high", "adj_low": "low", "adj_close": "close",
            "adj_volume": "volume"}
    written, parts, current = [], [], None

    def flush():
        if current is not None and parts:
            df = pd.concat(parts)
            df.index = pd.to_datetime(df.pop("date"))
            df.index.name = None
            df.sort_index().to_parquet(out / f"{current}.parquet")
            written.append(current)

    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        for chunk in pd.read_csv(f, usecols=["ticker", "date", *cols], chunksize=chunksize):
            chunk = chunk.rename(columns=cols)
            for ticker, part in chunk.groupby("ticker", sort=False):
                if ticker != current:
                    flush()
                    parts, current = [], ticker
                if keep is None or ticker in keep:
                    parts.append(part[["date", *COLUMNS]])
        flush()
    return written


# ---------------------------------------------------------------- Tiingo

class TiingoLimiter:
    """Keeps the free plan's limits using a CSV log of requests (time, symbol).

    `allow(symbol)` is False when a new symbol would exceed `per_month` unique symbols
    this calendar month, or a request would exceed `per_day`; `wait()` sleeps so that
    requests are at least `interval` seconds apart (50 an hour -> 72 s).
    """

    def __init__(self, log_path, per_month=500, per_day=1000, interval=75.0,
                 clock=time.time, sleep=time.sleep):
        self.log_path = Path(log_path)
        self.per_month, self.per_day, self.interval = per_month, per_day, interval
        self.clock, self.sleep = clock, sleep
        self.log = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                self.log = [(float(t), s) for t, s in csv.reader(f)]

    def _now(self):
        return dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc)

    def allow(self, symbol) -> bool:
        now = self._now()
        stamps = [(dt.datetime.fromtimestamp(t, dt.timezone.utc), s) for t, s in self.log]
        month = {s for d, s in stamps if (d.year, d.month) == (now.year, now.month)}
        today = [s for d, s in stamps if d.date() == now.date()]
        if symbol not in month and len(month) >= self.per_month:
            return False
        return len(today) < self.per_day

    def wait(self):
        if self.log:
            gap = self.clock() - self.log[-1][0]
            if gap < self.interval:
                self.sleep(self.interval - gap)

    def record(self, symbol):
        self.log.append((self.clock(), symbol))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{self.log[-1][0]:.0f}", symbol])


def parse_tiingo_csv(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    df.index = pd.to_datetime(df.pop("date")).dt.tz_localize(None)
    df.index.name = None
    df = df[["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]]
    df.columns = COLUMNS
    return df.sort_index()


def tiingo_download(symbols, out_dir, limiter, token_file=TIINGO_TOKEN, start="1990-01-01",
                    opener=urllib.request.urlopen, max_new=None) -> dict:
    """Fetch each Tiingo symbol not yet saved to `out_dir/<symbol>.parquet`.

    Stops at the limiter's limits or after `max_new` fetches. Returns
    {symbol: 'ok' | 'empty' | 'error <code>'} for the symbols tried.
    """
    token = Path(token_file).read_text().strip()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    for sym in symbols:
        if (out / f"{sym}.parquet").exists() or (out / f"{sym}.missing").exists():
            continue
        if (max_new is not None and len(status) >= max_new) or not limiter.allow(sym):
            break
        limiter.wait()
        limiter.record(sym)
        try:
            data = _get(TIINGO_PRICES.format(ticker=sym, start=start),
                        {"Authorization": f"Token {token}"}, opener)
        except urllib.error.HTTPError as e:
            status[sym] = f"error {e.code}"
            if e.code == 404:
                (out / f"{sym}.missing").write_text("404\n")
            if e.code == 429:
                break
            continue
        df = parse_tiingo_csv(data) if data.strip() else pd.DataFrame(columns=COLUMNS)
        if df.empty:
            (out / f"{sym}.missing").write_text("empty\n")
            status[sym] = "empty"
            continue
        df.to_parquet(out / f"{sym}.parquet")
        status[sym] = "ok"
    return status


def load_tiingo_list(data: bytes) -> pd.DataFrame:
    """Tiingo's supported-ticker list (zip of CSV) as a DataFrame keyed by upper-case ticker."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), keep_default_na=False)
    df["ticker"] = df["ticker"].str.upper()
    return df


# ---------------------------------------------------------------- checks

def overlap_check(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Agreement of two price series on their common days: number of days, correlation
    of daily close-to-close returns and the median absolute return difference."""
    ra = a["close"].pct_change(fill_method=None)
    rb = b["close"].pct_change(fill_method=None)
    df = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(df) < 20:
        return {"days": len(df), "corr": float("nan"), "mad": float("nan")}
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {"days": len(df), "corr": corr, "mad": float(np.median(np.abs(x - y)))}


def span_coverage(df: pd.DataFrame, first, last, calendar: pd.DatetimeIndex) -> float:
    """Share of trading days (from `calendar`) between first and last that `df` has."""
    days = calendar[(calendar >= first) & (calendar <= last)]
    if len(days) == 0:
        return float("nan")
    return float(days.isin(df.index).mean())


def check(spans: pd.DataFrame, out_dir=OUT, hf_dir="data/scratch/tickers",
          min_corr=0.9, min_cover=0.9) -> pd.DataFrame:
    """One row per member ticker: which sources have it, coverage of the membership
    period, agreement between sources, and a status: ok, mismatch, partial or missing."""
    out = Path(out_dir)
    spy = None
    for src in ("tiingo", "wiki"):
        p = out / src / "SPY.parquet"
        if p.exists():
            spy = pd.read_parquet(p)
            break
    rows = []
    for ticker, (first, last) in spans.iterrows():
        rec = {"ticker": ticker, "member_first": first.date(), "member_last": last.date()}
        series = {}
        for src, sym in (("wiki", wiki_symbol(ticker)), ("tiingo", tiingo_symbol(ticker))):
            p = out / src / f"{sym}.parquet"
            if p.exists():
                series[src] = pd.read_parquet(p)
                rec[f"{src}_first"] = series[src].index.min().date()
                rec[f"{src}_last"] = series[src].index.max().date()
        hf = Path(hf_dir) / f"{ticker}_clean.parquet"
        if "tiingo" in series and "wiki" in series:
            c = overlap_check(series["tiingo"], series["wiki"])
            rec |= {"wiki_days": c["days"], "wiki_corr": c["corr"]}
        if "tiingo" in series and hf.exists():
            from .data import BarLoader
            daily = BarLoader(hf_dir).load(ticker, timeframe="1D")
            c = overlap_check(series["tiingo"], daily)
            rec |= {"hf_days": c["days"], "hf_corr": c["corr"]}
        if not series:
            rec["status"] = "missing"
        else:
            merged = pd.concat(series.values()).index.unique().sort_values()
            cal = spy.index if spy is not None else merged
            rec["cover"] = span_coverage(pd.DataFrame(index=merged), first, last, cal)
            corrs = [rec[k] for k in ("wiki_corr", "hf_corr") if k in rec and not np.isnan(rec[k])]
            if any(c < min_corr for c in corrs):
                rec["status"] = "mismatch"
            elif rec["cover"] < min_cover:
                rec["status"] = "partial"
            else:
                rec["status"] = "ok"
        rows.append(rec)
    return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------- CLI

def _spans(out: Path, start: str) -> pd.DataFrame:
    return member_spans(parse_members((out / "sp500_members.csv").read_text()), start)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["membership", "wiki", "tiingo", "check"])
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--start", default=START, help="first membership date of the universe")
    p.add_argument("--zip", default="data/kaggle/wiki/quandl-wiki-prices-us-equites.zip")
    p.add_argument("--max", type=int, default=None, help="tiingo: at most this many fetches")
    p.add_argument("--extra", nargs="*", default=["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"],
                   help="tiingo: also fetch these (ETFs, benchmarks)")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "membership":
        (out / "sp500_members.csv").write_bytes(_get(MEMBERS_URL))
        (out / "tiingo_supported.zip").write_bytes(_get(TIINGO_LIST_URL))
        print("saved", out / "sp500_members.csv", "and", out / "tiingo_supported.zip")
    elif args.action == "wiki":
        spans = _spans(out, args.start)
        names = wiki_extract(args.zip, out / "wiki", [wiki_symbol(t) for t in spans.index])
        print(f"wiki: wrote {len(names)} of {len(spans)} members")
    elif args.action == "tiingo":
        spans = _spans(out, args.start)
        listed = set(load_tiingo_list((out / "tiingo_supported.zip").read_bytes())["ticker"])
        wiki_done = set()
        for t, (first, last) in spans.iterrows():       # WIKI is complete if they left before it froze
            w = out / "wiki" / f"{wiki_symbol(t)}.parquet"
            if w.exists() and last < pd.Timestamp("2018-03-01"):
                wiki_done.add(t)
        syms = list(dict.fromkeys(args.extra + [tiingo_symbol(t) for t in spans.index
                                                if t not in wiki_done and tiingo_symbol(t) in listed]))
        limiter = TiingoLimiter(out / "tiingo_requests.csv")
        status = tiingo_download(syms, out / "tiingo", limiter, max_new=args.max)
        done = sum((out / "tiingo" / f"{s}.parquet").exists() for s in syms)
        print(f"tiingo: tried {len(status)} ({sum(v == 'ok' for v in status.values())} ok); "
              f"{done} of {len(syms)} wanted symbols saved")
        for s, v in status.items():
            if v != "ok":
                print(" ", s, v)
    else:
        table = check(_spans(out, args.start), out)
        table.to_csv(out / "coverage.csv")
        print(table["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
