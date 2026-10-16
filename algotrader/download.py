"""Download HF Data Library 1-minute bars ("clean" version) as parquet files.

The API key is read from a file (default ~/.config/hfdatalibrary/token) and is
never printed or written anywhere else.

Example:
    python -m algotrader.download --out data/scratch/tickers SPY QQQ AAPL
    python -m algotrader.download --out data/scratch/tickers --universe broad
"""
import argparse
import time
import urllib.request
from pathlib import Path

API = "https://api.hfdatalibrary.com/v1/bars/{ticker}?version=clean"
DEFAULT_TOKEN = Path.home() / ".config" / "hfdatalibrary" / "token"


def download(tickers, out_dir, token_file=DEFAULT_TOKEN, force=False, pause=0.7,
             opener=urllib.request.urlopen) -> list:
    """Fetch each ticker to `out_dir/<TICKER>_clean.parquet`; returns tickers fetched.

    Existing files are skipped unless `force`. `pause` seconds between requests keeps
    under the provider's 100 downloads per minute.
    """
    key = Path(token_file).read_text().strip()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fetched = []
    for ticker in tickers:
        target = out / f"{ticker}_clean.parquet"
        if target.exists() and not force:
            continue
        # The API rejects urllib's default User-Agent with 403.
        req = urllib.request.Request(API.format(ticker=ticker),
                                     headers={"X-API-Key": key, "User-Agent": "algotrader/0.1"})
        with opener(req) as resp:
            data = resp.read()
        if not data.startswith(b"PAR1"):
            raise RuntimeError(f"{ticker}: response is not a parquet file")
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        fetched.append(ticker)
        time.sleep(pause)
    return fetched


def main(argv=None):
    from .universe import UNIVERSES

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*")
    p.add_argument("--universe", choices=sorted(UNIVERSES),
                   help="also fetch every ticker of this universe (core, broad)")
    p.add_argument("--out", default="data/scratch/tickers")
    p.add_argument("--token-file", default=str(DEFAULT_TOKEN))
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args(argv)
    tickers = list(dict.fromkeys(args.tickers + (UNIVERSES[args.universe] if args.universe else [])))
    if not tickers:
        p.error("give tickers or --universe")
    fetched = download(tickers, args.out, args.token_file, args.force)
    print(f"downloaded {len(fetched)} of {len(tickers)} ({len(tickers) - len(fetched)} already present)")


if __name__ == "__main__":
    main()
