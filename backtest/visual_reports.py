#!/usr/bin/env python3
"""
Generate advanced visual reports for SMA 20/50 backtest results.
- Equity curves
- Drawdown plots
- Monthly returns heatmap (using matplotlib)
- Trade duration analysis
- Rolling Sharpe
Saves plots to output/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results'
OUTPUT_DIR = CACHE_DIR / 'scratch' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_summary():
    summary_file = RESULTS_DIR / 'summary_with_commission_all.csv'
    if not summary_file.exists():
        # try alternative
        summary_file = RESULTS_DIR / 'summary_with_costs.csv'
    df = pd.read_csv(summary_file)
    return df

def load_equity(ticker):
    # Find equity file for ticker: any file that starts with ticker and ends with _equity.csv
    pattern = f"{ticker}*_equity.csv"
    matches = list(RESULTS_DIR.glob(pattern))
    print(f"DEBUG: ticker={ticker}, pattern={pattern}, matches={matches}")
    if not matches:
        raise FileNotFoundError(f"No equity file for {ticker} matching {pattern}")
    # Prefer the one without extra suffix like _5min? but any will do.
    # Take the first match
    f = matches[0]
    print(f"DEBUG: loading equity from {f}")
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df['equity']

def load_trades(ticker):
    pattern = f"{ticker}*_trades.csv"
    matches = list(RESULTS_DIR.glob(pattern))
    print(f"DEBUG: ticker={ticker}, pattern={pattern}, matches={matches}")
    if not matches:
        raise FileNotFoundError(f"No trades file for {ticker} matching {pattern}")
    f = matches[0]
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_equity_curves(tickers, max_n=4):
    plt.figure(figsize=(12, 6))
    for i, ticker in enumerate(tickers[:max_n]):
        equity = load_equity(ticker)
        # normalize to starting capital for comparison
        equity_norm = equity / equity.iloc[0] * 10000
        plt.plot(equity_norm.index, equity_norm.values, label=ticker, linewidth=1)
    plt.title('Equity Curves (Normalized to $10,000)')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'equity_curves.png', dpi=150)
    plt.close()

def plot_drawdown(tickers, max_n=4):
    plt.figure(figsize=(12, 6))
    for i, ticker in enumerate(tickers[:max_n]):
        equity = load_equity(ticker)
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        plt.plot(drawdown.index, drawdown.values, label=ticker, linewidth=1)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'drawdown.png', dpi=150)
    plt.close()

def plot_monthly_returns_heatmap(ticker):
    equity = load_equity(ticker)
    # resample to monthly returns
    monthly = equity.resample('1M').last().pct_change().fillna(0)
    # create a DataFrame with year and month
    monthly.index = monthly.index.to_period('M')
    # pivot for heatmap: years as rows, months as columns
    data = monthly.to_frame(name='returns')
    data['year'] = data.index.year
    data['month'] = data.index.month
    pivot = data.pivot(index='year', columns='month', values='returns')
    # fill missing with NaN
    plt.figure(figsize=(10, 8))
    # Convert to numpy array for imshow
    data_vals = pivot.values
    # Mask NaN values
    masked_vals = np.ma.masked_invalid(data_vals)
    # Plot
    im = plt.imshow(masked_vals, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    plt.colorbar(im, label='Monthly Return')
    # Set ticks
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.xticks(range(len(pivot.columns)), [f'{m:02d}' for m in pivot.columns])
    plt.title(f'Monthly Returns Heatmap for {ticker}')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_monthly_returns_heatmap.png', dpi=150)
    plt.close()

def plot_trade_duration(ticker):
    trades = load_trades(ticker)
    # compute duration for each round-trip trade
    entries = trades[trades['type'] == 'entry'].copy()
    exits = trades[trades['type'] == 'exit'].copy()
    if entries.empty or exits.empty:
        print(f"No complete trades for {ticker}")
        return
    # assume they are in order; we'll pair by index
    entries = entries.reset_index(drop=True)
    exits = exits.reset_index(drop=True)
    # take min length
    n = min(len(entries), len(exits))
    entries = entries.iloc[:n]
    exits = exits.iloc[:n]
    duration = (exits['timestamp'] - entries['timestamp']).dt.total_seconds() / 3600  # hours
    plt.figure(figsize=(10, 6))
    plt.hist(duration, bins=50, edgecolor='black', alpha=0.7)
    plt.title(f'Trade Duration Distribution for {ticker}')
    plt.xlabel('Duration (hours)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_trade_duration.png', dpi=150)
    plt.close()

def plot_rolling_sharpe(ticker, window=252):  # window in trading days
    equity = load_equity(ticker)
    # compute daily returns
    daily = equity.resample('1D').last().ffill()
    ret = daily.pct_change().fillna(0)
    # rolling Sharpe (annualized)
    rolling_mean = ret.rolling(window=window).mean()
    rolling_std = ret.rolling(window=window).std()
    sharpe = rolling_mean / rolling_std * np.sqrt(252)
    plt.figure(figsize=(12, 6))
    plt.plot(sharpe.index, sharpe.values, label='Rolling Sharpe (1Y)')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Rolling Sharpe Ratio ({window}-day) for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Sharpe')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{ticker}_rolling_sharpe.png', dpi=150)
    plt.close()

def main():
    summary = load_summary()
    print(f'Loaded summary with {len(summary)} tickers')
    # Select top and bottom performers by total_return
    sorted_df = summary.sort_values('total_return', ascending=False)
    top_tickers = sorted_df.head(2)['ticker'].tolist()
    bottom_tickers = sorted_df.tail(2)['ticker'].tolist()
    selected = list(dict.fromkeys(top_tickers + bottom_tickers))  # unique preserving order
    print(f'Selected tickers for visuals: {selected}')
    # Generate plots
    plot_equity_curves(selected)
    plot_drawdown(selected)
    for ticker in selected:
        try:
            plot_monthly_returns_heatmap(ticker)
        except Exception as e:
            print(f'Failed monthly heatmap for {ticker}: {e}')
        try:
            plot_trade_duration(ticker)
        except Exception as e:
            print(f'Failed trade duration for {ticker}: {e}')
        try:
            plot_rolling_sharpe(ticker)
        except Exception as e:
            print(f'Failed rolling Sharpe for {ticker}: {e}')
    print(f'Plots saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()