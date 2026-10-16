#!/usr/bin/env python3
"""
Debug version of sma_backtest_txn to see what's causing the error.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
TICKERS_DIR = CACHE_DIR / 'scratch' / 'tickers'
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results_txn'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAST = 20
SLOW = 50
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0

SLIPPAGE_FACTOR = 0.1
PARTICIPATION_LIMIT = 0.1

def compute_stats(equity_curve, trades, ticker):
    returns = equity_curve.pct_change().fillna(0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25 if days > 0 else 1
    if years > 0 and (1 + total_return) > 0:
        annual_return = (1 + total_return) ** (1/years) - 1
    else:
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
    volume = df['Volume']
    # Compute SMAs
    df['sma_fast'] = close.rolling(window=FAST, min_periods=FAST).mean()
    df['sma_slow'] = close.rolling(window=SLOW, min_periods=SLOW).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df['position'] = df['signal'].diff()
    # Latency: next bar's open price (1 bar)
    df['next_open'] = open_price.shift(-1)
    # DEBUG: print types
    # Slippage: proportional to spread (high-low) as fraction of close
    spread = (high - low) / close  # fractional spread
    slippage = SLIPPAGE_FACTOR * spread  # fractional slippage
    # Apply slippage to execution price: buy at higher price, sell at lower
    df['exec_price_buy'] = df['next_open'] * (1 + slippage)
    df['exec_price_sell'] = df['next_open'] * (1 - slippage)
    # For last bar where next_open is NaN, fallback to close
    df['exec_price_buy'] = df['exec_price_buy'].fillna(close * (1 + slippage))
    df['exec_price_sell'] = df['exec_price_sell'].fillna(close * (1 - slippage))
    equity = INITIAL_CAPITAL
    position = 0  # number of shares
    entry_price = 0.0
    trades = []
    equity_curve = []
    for idx, row in df.iterrows():
        # Equity before execution (using next_open price for consistency)
        price_mid = row['next_open'] if not pd.isna(row['next_open']) else close.loc[idx]
        equity_curve.append({'timestamp': idx, 'equity': equity + position * price_mid})
        # Entry signal
        if row['position'] == 1 and position == 0:
            price = row['exec_price_buy']
            # Estimate commission per share
            shares_gross = equity / (price + COMMISSION_PER_SHARE)
            if shares_gross * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                commission_per_share = COMMISSION_PER_SHARE
                shares = shares_gross
            else:
                shares = (equity - COMMISSION_MIN_PER_TRADE) / price if equity > COMMISSION_MIN_PER_TRADE else 0
                commission_per_share = COMMISSION_MIN_PER_TRADE / shares if shares > 0 else 0
            # Partial fill limit based on volume
            max_shares_by_volume = row['Volume'] * PARTICIPATION_LIMIT
            shares = min(shares, max_shares_by_volume)
            if shares <= 0:
                continue
            cost = shares * price + shares * commission_per_share
            if cost > equity:
                shares = equity / (price + commission_per_share)
                cost = shares * price + shares * commission_per_share
            position = shares
            equity -= cost
            entry_price = price
            trades.append({
                'type': 'entry', 'timestamp': idx, 'price': row['next_open'] if not pd.isna(row['next_open']) else close.loc[idx],
                'shares': shares, 'commission': shares * commission_per_share,
                'pnl': 0, 'slippage': slippage.loc[idx]
            })
        # Exit signal
        elif row['position'] == -1 and position > 0:
            price = row['exec_price_sell']
            shares_gross = position
            if shares_gross * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                commission_per_share = COMMISSION_PER_SHARE
            else:
                commission_per_share = COMMISSION_MIN_PER_TRADE / shares_gross if shares_gross > 0 else 0
            proceeds = shares_gross * price - shares_gross * commission_per_share
            equity += proceeds
            pnl = (price - entry_price) * shares_gross - shares_gross * commission_per_share
            trades.append({
                'type': 'exit', 'timestamp': idx, 'price': row['next_open'] if not pd.isna(row['next_open']) else close.loc[idx],
                'shares': shares_gross, 'commission': shares_gross * commission_per_share,
                'pnl': pnl, 'slippage': slippage.loc[idx]
            })
            position = 0
            entry_price = 0.0
    # Close any remaining position at end using close price (no next bar)
    if position > 0:
        price = close.iloc[-1]
        slippage_last = slippage.iloc[-1] if len(slippage) > 0 else 0
        price = price * (1 - slippage_last)  # sell slippage
        if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
            commission_per_share = COMMISSION_PER_SHARE
        else:
            commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
        proceeds = position * price - position * commission_per_share
        equity += proceeds
        pnl = (price - entry_price) * position - position * commission_per_share
        trades.append({
            'type': 'exit', 'timestamp': df.index[-1], 'price': close.iloc[-1],
            'shares': position, 'commission': position * commission_per_share,
            'pnl': pnl, 'slippage': slippage_last
        })
        position = 0
    equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
    stats = compute_stats(equity_curve_df['equity'], trades, ticker_file.stem.replace('_clean', ''))
    return stats, trades, equity_curve_df

def main():
    all_stats = []
    for parquet_file in TICKERS_DIR.glob('*_clean.parquet'):
        print(f'Processing {parquet_file.name}')
        try:
            stats, trades, equity_curve = backtest_ticker(parquet_file)
            all_stats.append(stats)
            equity_curve.to_csv(RESULTS_DIR / f"{parquet_file.stem}_equity_txn_debug.csv")
            pd.DataFrame(trades).to_csv(RESULTS_DIR / f"{parquet_file.stem}_trades_txn_debug.csv", index=False)
        except Exception as e:
            print(f'Error processing {parquet_file.name}: {e}')
            import traceback
            traceback.print_exc()
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(RESULTS_DIR / 'summary_txn_debug_all.csv', index=False)
        print('Saved summary to results_txn_debug/summary_txn_debug_all.csv')
        print('\n=== Aggregate Results (Txn Cost Debug) ===')
        print(f'Mean total return: {stats_df["total_return"].mean():.2%}')
        print(f'Median total return: {stats_df["total_return"].median():.2%}')
        print(f'Mean Sharpe: {stats_df["sharpe"].mean():.2f}')
        print(f'Mean max drawdown: {stats_df["max_drawdown"].mean():.2%}')
    else:
        print('No stats generated')

if __name__ == '__main__':
    main()