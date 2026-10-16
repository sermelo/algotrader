#!/usr/bin/env python3
"""
Walk‑forward / out‑of‑sample validation and simple parameter sensitivity (demo on one ticker, limited data).
- Splits data into training (first 2 years) and testing (remaining).
- Evaluates default SMA (20/50) on both train and test.
- Also evaluates two alternative SMA settings (10/30 and 30/60) on training set to show sensitivity.
- Reports performance metrics for each.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Configuration
CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
TICKERS_DIR = CACHE_DIR / 'scratch' / 'tickers'
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared parameters
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
MAX_POSITION_FRACTION = 0.20
STOP_LOSS_PCT = 0.02
ATR_PERIOD = 14
ATR_THRESHOLD_PCT = 0.005

def compute_atr(df, period=ATR_PERIOD):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

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

def resample_to_5min(df):
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df_5min = df.resample('5min').apply(ohlc_dict).dropna()
    return df_5min

def backtest_ticker_from_df(df, fast=20, slow=50, use_5min=False):
    # df already has datetime index and columns Open, High, Low, Close, Volume
    if use_5min:
        df = resample_to_5min(df)
        if df.empty:
            return {'ticker': 'UNKNOWN', 'total_return':0, 'annual_return':0, 'sharpe':np.nan, 'max_drawdown':0, 'trades':0, 'win_rate':np.nan, 'avg_win':0, 'avg_loss':0, 'profit_factor':np.nan, 'final_equity':INITIAL_CAPITAL}, [], pd.DataFrame()
    # Indicators
    close = df['Close']
    high = df['High']
    low = df['Low']
    df['sma_fast'] = close.rolling(window=fast, min_periods=fast).mean()
    df['sma_slow'] = close.rolling(window=slow, min_periods=slow).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df['position'] = df['signal'].diff()
    # ATR
    df['atr'] = compute_atr(df, ATR_PERIOD)
    # Trading variables
    equity = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    stop_loss_price = 0.0
    trades = []
    equity_curve = []
    for idx, row in df.iterrows():
        price = row['Close']
        atr = row['atr'] if not pd.isna(row['atr']) else 0.0
        equity_curve.append({'timestamp': idx, 'equity': equity + position * price})
        # Stop loss exit
        if position > 0 and stop_loss_price > 0 and price <= stop_loss_price:
            revenue = position * price
            if position * 0.005 >= 1.0:
                commission = position * 0.005
            else:
                commission = 1.0
            proceeds = revenue - commission
            equity += proceeds
            pnl = (price - entry_price) * position - commission
            trades.append({'type': 'exit_stop', 'timestamp': idx, 'price': price, 'shares': position, 'commission': commission, 'pnl': pnl})
            position = 0
            entry_price = 0
            stop_loss_price = 0.0
            continue
        # Entry signal
        if row['position'] == 1:  # enter long
            if position == 0:
                if atr <= ATR_THRESHOLD_PCT * price:
                    continue
                if equity <= 1.0:
                    continue
                risk_amount = equity * RISK_PER_TRADE
                stop_loss = price * (1.0 - STOP_LOSS_PCT)
                if price <= stop_loss:
                    continue
                shares_risk = risk_amount / (price - stop_loss)
                shares_max = (equity * MAX_POSITION_FRACTION) / price
                shares = min(shares_risk, shares_max)
                if shares <= 0:
                    continue
                if shares * 0.005 >= 1.0:
                    commission = shares * 0.005
                else:
                    commission = 1.0
                cost = shares * price + commission
                if cost > equity:
                    continue
                position = shares
                equity -= cost
                entry_price = price
                stop_loss_price = price * (1.0 - STOP_LOSS_PCT)
                trades.append({'type': 'entry', 'timestamp': idx, 'price': price, 'shares': shares, 'commission': commission, 'pnl': 0})
        # Exit signal (opposite SMA crossover)
        elif row['position'] == -1:  # exit long
            if position > 0:
                revenue = position * price
                if position * 0.005 >= 1.0:
                    commission = position * 0.005
                else:
                    commission = 1.0
                proceeds = revenue - commission
                equity += proceeds
                pnl = (price - entry_price) * position - commission
                trades.append({'type': 'exit_signal', 'timestamp': idx, 'price': price, 'shares': position, 'commission': commission, 'pnl': pnl})
                position = 0
                entry_price = 0
                stop_loss_price = 0.0
    # Close remaining
    if position > 0:
        price = df.iloc[-1]['Close']
        revenue = position * price
        if position * 0.005 >= 1.0:
            commission = position * 0.005
        else:
            commission = 1.0
        proceeds = revenue - commission
        equity += proceeds
        pnl = (price - entry_price) * position - commission
        trades.append({'type': 'exit_final', 'timestamp': df.index[-1], 'price': price, 'shares': position, 'commission': commission, 'pnl': pnl})
        position = 0
        entry_price = 0
        stop_loss_price = 0.0
    equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
    stats = compute_stats(equity_curve_df['equity'], trades, 'UNKNOWN')
    return stats, trades, equity_curve_df

def walk_forward(ticker_file):
    df = pd.read_parquet(ticker_file)
    df = df.sort_values('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    # Limit to last 200k rows for speed
    if len(df) > 200000:
        df = df.iloc[-200000:]
    # Define split: first 2 years as train, rest as test (within the limited data)
    start_date = df.index.min()
    end_date = df.index.max()
    split_date = start_date + pd.Timedelta(days=730)  # approx 2 years
    if split_date >= end_date:
        # Not enough data, use 80/20 split
        split_date = df.index[int(0.8 * len(df))]
    train = df.loc[start_date:split_date].copy()
    test = df.loc[split_date:].copy()
    print(f'  Train period: {train.index.min()} to {train.index.max()} ({len(train)} bars)')
    print(f'  Test period:  {test.index.min()} to {test.index.max()} ({len(test)} bars)')
    # Evaluate default SMA (20/50) on train and test
    stats_train_default, _, _ = backtest_ticker_from_df(train, fast=20, slow=50, use_5min=False)
    stats_test_default, _, _ = backtest_ticker_from_df(test, fast=20, slow=50, use_5min=False)
    # Evaluate two alternative SMA settings on train (sensitivity)
    stats_train_alt1, _, _ = backtest_ticker_from_df(train, fast=10, slow=30, use_5min=False)
    stats_train_alt2, _, _ = backtest_ticker_from_df(train, fast=30, slow=60, use_5min=False)
    # Also evaluate alternatives on test for completeness
    stats_test_alt1, _, _ = backtest_ticker_from_df(test, fast=10, slow=30, use_5min=False)
    stats_test_alt2, _, _ = backtest_ticker_from_df(test, fast=30, slow=60, use_5min=False)
    return {
        'train_default': stats_train_default,
        'test_default': stats_test_default,
        'train_alt1': stats_train_alt1,
        'train_alt2': stats_train_alt2,
        'test_alt1': stats_test_alt1,
        'test_alt2': stats_test_alt2
    }

def main():
    # Process only the first ticker for demo
    files = list(TICKERS_DIR.glob('*_clean.parquet'))
    if not files:
        print('No ticker files found')
        return
    f = files[0]
    print(f'Processing {f.name} (limited to last 200k rows)')
    try:
        result = walk_forward(f)
        summary = {
            'ticker': f.name.replace('_clean.parquet', ''),
            'train_return_default': result['train_default']['total_return'],
            'train_sharpe_default': result['train_default']['sharpe'],
            'test_return_default': result['test_default']['total_return'],
            'test_sharpe_default': result['test_default']['sharpe'],
            'train_return_alt1': result['train_alt1']['total_return'],
            'train_sharpe_alt1': result['train_alt1']['sharpe'],
            'train_return_alt2': result['train_alt2']['total_return'],
            'train_sharpe_alt2': result['train_alt2']['sharpe'],
            'test_return_alt1': result['test_alt1']['total_return'],
            'test_sharpe_alt1': result['test_alt1']['sharpe'],
            'test_return_alt2': result['test_alt2']['total_return'],
            'test_sharpe_alt2': result['test_alt2']['sharpe']
        }
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(RESULTS_DIR / 'walkforward_summary.csv', index=False)
        print('Saved walk‑forward summary to results/walkforward_summary.csv')
        print(df_summary.to_string(index=False))
    except Exception as e:
        print(f'Error processing {f.name}: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()