#!/usr/bin/env python3
import sys
sys.path.append('/home/hermes/.hermes/scripts/backtest')
from sma_backtest_voladj_atr4 import backtest_ticker
from pathlib import Path
import os
cache = os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache')
tickers_dir = Path(cache) / 'scratch' / 'tickers'
files = list(tickers_dir.glob('*_clean.parquet'))[:5]
all_stats = []
for f in files:
    print(f'Processing {f.name}')
    try:
        stats, trades, equity = backtest_ticker(f)
        all_stats.append(stats)
        print(f'  Sharpe: {stats.get("sharpe", "NaN")}, Return: {stats.get("total_return", 0):.2%}, Trades: {stats.get("trades", 0)}')
    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()
if all_stats:
    import pandas as pd
    df = pd.DataFrame(all_stats)
    print('\\nSummary:')
    print(df[['ticker','total_return','sharpe','trades']])
    print(f'\\nMean total return: {df["total_return"].mean():.2%}')
    print(f'Mean Sharpe: {df["sharpe"].mean():.2f}')