#!/usr/bin/env python3
"""
SMA 20/50 Crossover Backtest on 1‑minute data.
- Reads cleaned parquet files from $HERMES_CACHE/scratch/tickers/
- Computes SMA20 and SMA50 on close price.
- Enters long when SMA20 crosses above SMA50, exits when SMA20 crosses below.
- Assumes zero‑size slippage, commission per share ($0.005, min $1 per trade).
- Outputs per‑ticker statistics to CSV in $HERMES_CACHE/scratch/results/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
TICKERS_DIR = CACHE_DIR / 'scratch' / 'tickers'
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAST = 20
SLOW = 50
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0  # per ticker for simplicity

SLIPPAGE_PER_SHARE = 0.001   # slippage per share (absolute)
LATENCY_BARS = 1             # execute at next bar's open after signal

def compute_stats(equity_curve, trades, ticker):
    """Calculate performance metrics."""
    returns = equity_curve.pct_change().fillna(0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25 if days > 0 else 1
    if years > 0 and (1 + total_return) > 0:
        annual_return = (1 + total_return) ** (1/years) - 1
    else:
        # Fallback: approximate annual return as total_return / years (simple)
        annual_return = total_return / years if years > 0 else total_return
    if len(returns) > 1:
        avg_minutes = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / len(returns) / 60
        minutes_per_year = 252 * 6.5 * 60
        periods_per_year = minutes_per_year / avg_minutes if avg_minutes > 0 else 0
    else:
        periods_per_year = 0
    if returns.std() > 0 and periods_per_year > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = np.nan
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    win_trades = [t for t in trades if t.get('pnl', 0) > 0]
    loss_trades = [t for t in trades if t.get('pnl', 0) <= 0]
    win_rate = len(win_trades) / len(trades) if trades else np.nan
    avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
    avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0
    profit_factor = abs(sum([t['pnl'] for t in win_trades]) / sum([t['pnl'] for t in loss_trades])) if loss_trades and sum([t['pnl'] for t in loss_trades]) != 0 else np.nan
    stats = {
        'ticker': ticker,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_equity': equity_curve.iloc[-1],
    }
    return stats

def backtest_ticker(ticker_file):
    df = pd.read_parquet(ticker_file)
    df = df.sort_values('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    close = df['Close']
    open_price = df['Open']
    high = df['High']
    low = df['Low']
    df['sma_fast'] = close.rolling(window=FAST, min_periods=FAST).mean()
    df['sma_slow'] = close.rolling(window=SLOW, min_periods=SLOW).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df['position'] = df['signal'].diff()
    # Latency: next bar's open price
    df['next_open'] = open_price.shift(-1)
    # Simulate trading
    equity = INITIAL_CAPITAL
    position = 0  # number of shares
    entry_price = 0.0
    trades = []
    equity_curve = []
    for idx, row in df.iterrows():
        price_close = row['Close']
        # Determine execution price based on latency
        if pd.isna(row['next_open']):
            exec_price = price_close
        else:
            exec_price = row['next_open']
        equity_curve.append({'timestamp': idx, 'equity': equity + position * exec_price})
        if row['position'] == 1:  # enter long
            if position == 0:
                # Compute max shares we can buy after commission and slippage
                # Slippage per share added to execution price for buys
                price_with_slippage = exec_price + SLIPPAGE_PER_SHARE
                if equity <= 1.0:
                    # Not enough to cover even the $1 minimum commission
                    continue
                # Regime 1: commission = shares * 0.005 (applies when shares >= 200)
                shares1 = equity / (price_with_slippage + 0.005)
                if shares1 >= 200:
                    shares = shares1
                    commission = shares * 0.005
                else:
                    # Regime 2: commission = 1.0 (applies when shares < 200)
                    shares2 = (equity - 1.0) / price_with_slippage
                    if shares2 < 200 and shares2 > 0:
                        shares = shares2
                        commission = 1.0
                    else:
                        # Not enough capital to buy any shares after commission
                        continue
                if shares <= 0:
                    continue
                cost = shares * price_with_slippage + commission
                position = shares
                equity -= cost
                entry_price = price_with_slippage  # entry price includes slippage
                trades.append({'type': 'entry', 'timestamp': idx, 'price': exec_price, 'shares': shares, 'commission': commission, 'pnl': 0, 'slippage': SLIPPAGE_PER_SHARE})
        elif row['position'] == -1:  # exit long
            if position > 0:
                # Determine execution price for exit
                if pd.isna(row['next_open']):
                    exec_price = price_close
                else:
                    exec_price = row['next_open']
                price_with_slippage = exec_price - SLIPPAGE_PER_SHARE  # slippage reduces proceeds for sells
                if position * 0.005 >= 1.0:
                    commission = position * 0.005
                else:
                    commission = 1.0
                proceeds = position * price_with_slippage - commission
                equity += proceeds
                pnl = (price_with_slippage - entry_price) * position - commission
                trades.append({'type': 'exit', 'timestamp': idx, 'price': exec_price, 'shares': position, 'commission': commission, 'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE})
                position = 0
                entry_price = 0.0
    if position > 0:
        # Close any remaining position at end using close price (no next bar)
        price_close = df.iloc[-1]['Close']
        exec_price = price_close  # no next bar
        price_with_slippage = exec_price - SLIPPAGE_PER_SHARE
        if position * 0.005 >= 1.0:
            commission = position * 0.005
        else:
            commission = 1.0
        proceeds = position * price_with_slippage - commission
        equity += proceeds
        pnl = (price_with_slippage - entry_price) * position - commission
        trades.append({'type': 'exit', 'timestamp': df.index[-1], 'price': exec_price, 'shares': position, 'commission': commission, 'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE})
        position = 0
        entry_price = 0.0
    equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
    # If no trades, we still want to compute stats (equity curve unchanged)
    stats = compute_stats(equity_curve_df['equity'], trades, ticker_file.stem.replace('_clean', ''))
    # Debug: if no trades, print a warning
    if len(trades) == 0:
        print(f'Warning: {ticker_file.name} generated zero trades.')
    return stats, trades, equity_curve_df
def main():
    all_stats = []
        except Exception as e:
    files = list(TICKERS_DIR.glob('*_clean.parquet'))[:5]
    for parquet_file in files:
        print(f'Processing {parquet_file.name}')
        try:
            stats, trades, equity_curve = backtest_ticker(parquet_file)
            all_stats.append(stats)
            equity_curve.to_csv(RESULTS_DIR / f"{parquet_file.stem}_equity.csv")
            pd.DataFrame(trades).to_csv(RESULTS_DIR / f"{parquet_file.stem}_trades.csv", index=False)
        except Exception as e:
            print(f'Error processing {parquet_file.name}: {e}')
            print(f'Error processing {parquet_file.name}: {e}')
        stats_df.to_csv(RESULTS_DIR / 'summary_with_commission_all.csv', index=False)
        print('Saved summary to results/summary_with_commission_all.csv')
    else:
        print('No stats generated')

if __name__ == '__main__':
    main()