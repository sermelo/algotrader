"""Command line entry point.

Examples:
    python -m algotrader.run --data-dir data/scratch/tickers \\
        --tickers ADM AMC AMGN --timeframe 5min --out data/results/summary.csv

    # decision tree per ticker: fit before --train-end, backtest from it on
    python -m algotrader.run --data-dir data/scratch/tickers --strategy tree \\
        --train-end 2024-01-01 --timeframe 1min
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
from .tree import DecisionTreeStrategy


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with <TICKER>_clean.parquet")
    p.add_argument("--tickers", nargs="*", help="default: every file in data-dir")
    p.add_argument("--timeframe", default="5min", help="bar size, pandas rule (default 5min)")
    p.add_argument("--strategy", choices=["sma", "tree"], default="sma",
                   help="sma: SMA crossover (default); tree: decision tree (needs --train-end)")
    p.add_argument("--train-end", help="tree: fit on bars before this date, backtest the rest")
    p.add_argument("--max-depth", type=int, default=4, help="tree: depth (default 4)")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--trend-filter", type=int, default=None, help="only long above this SMA")
    p.add_argument("--atr-period", type=int, default=14, help="ATR window in bars (default 14)")
    p.add_argument("--atr-mult", type=float, default=2.0, help="stop distance in ATRs (default 2)")
    p.add_argument("--risk", type=float, default=0.01, help="equity fraction risked per trade")
    p.add_argument("--max-position", type=float, default=0.10, help="max equity fraction per position")
    p.add_argument("--commission", type=float, default=0.005, help="$ per share (default 0.005)")
    p.add_argument("--min-commission", type=float, default=1.0, help="$ minimum per order (default 1)")
    p.add_argument("--slippage", type=float, default=0.0, help="$ per share per side")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--out", help="write summary CSV here")
    args = p.parse_args(argv)

    loader = BarLoader(args.data_dir)
    tickers = args.tickers or loader.tickers()
    if not tickers:
        p.error(f"no *_clean.parquet files in {args.data_dir}")
    if args.strategy == "tree" and not args.train_end:
        p.error("--strategy tree needs --train-end")
    train_end = pd.Timestamp(args.train_end) if args.strategy == "tree" else None

    def make_strategy():
        if args.strategy == "sma":
            return SmaCrossStrategy(args.fast, args.slow, args.atr_period, args.atr_mult,
                                    args.trend_filter)
        return DecisionTreeStrategy(args.max_depth, args.atr_period, args.atr_mult)

    try:
        make_strategy()                     # validate parameters before loading data
    except ValueError as e:
        p.error(str(e))
    costs = CostModel(args.commission, args.min_commission, args.slippage)
    sizer = PositionSizer(args.risk, args.max_position)

    rows = []
    for ticker in tickers:
        bars = loader.load(ticker, args.timeframe)
        strategy = make_strategy()
        if train_end is not None:           # per-ticker tree, fitted on the earlier period only
            train, bars = bars[bars.index < train_end], bars[bars.index >= train_end]
            if train.empty or bars.empty:
                p.error(f"{ticker}: no bars on one side of --train-end {args.train_end}")
            strategy.fit(train)
        result = Backtester(strategy, costs, sizer, args.capital).run(bars)
        row = {"ticker": ticker, **Metrics(result, bars).summary()}
        rows.append(row)
        print(f"{ticker}: {row['total_return']:+.2%} ({row['trades']} trades, "
              f"in market {row['exposure']:.0%}, B&H {row['buy_and_hold']:+.2%})")
    summary = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
    return summary


if __name__ == "__main__":
    main()
