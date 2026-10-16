"""Command line entry point.

Example:
    python -m algotrader.run --data-dir data/scratch/tickers \\
        --tickers ADM AMC AMGN --timeframe 5min --out data/results/summary.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from .costs import CostModel
from .data import BarLoader
from .engine import Backtester
from .metrics import Metrics
from .sizing import PositionSizer
from .strategy import SmaCrossStrategy


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with <TICKER>_clean.parquet")
    p.add_argument("--tickers", nargs="*", help="default: every file in data-dir")
    p.add_argument("--timeframe", default="5min", help="bar size, pandas rule (default 5min)")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--trend-filter", type=int, default=None, help="only long above this SMA")
    p.add_argument("--risk", type=float, default=0.01, help="equity fraction risked per trade")
    p.add_argument("--max-position", type=float, default=0.10, help="max equity fraction per position")
    p.add_argument("--slippage", type=float, default=0.0, help="$ per share per side")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--out", help="write summary CSV here")
    args = p.parse_args(argv)

    loader = BarLoader(args.data_dir)
    engine = Backtester(SmaCrossStrategy(args.fast, args.slow, trend_filter=args.trend_filter),
                        CostModel(slippage_per_share=args.slippage),
                        PositionSizer(args.risk, args.max_position), args.capital)
    rows = []
    for ticker in args.tickers or loader.tickers():
        bars = loader.load(ticker, args.timeframe)
        rows.append({"ticker": ticker, **Metrics(engine.run(bars), bars).summary()})
        print(f"{ticker}: {rows[-1]['total_return']:+.2%} "
              f"({rows[-1]['trades']} trades, B&H {rows[-1]['buy_and_hold']:+.2%})")
    summary = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
    return summary


if __name__ == "__main__":
    main()
