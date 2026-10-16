#!/usr/bin/env python3
"""
SMA 20/50 Crossover Backtest on 5‑minute data (resampled from 1‑minute).
Includes risk‑management rules: fixed‑fraction position sizing, max position size,
stop‑loss, volatility filter using ATR.
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

# Strategy parameters
FAST = 20          # fast SMA period
SLOW = 50          # slow SMA period
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0

# Risk management parameters
RISK_PER_TRADE = 0.01          # 1% of equity at risk per trade
MAX_POSITION_FRACTION = 0.20   # max 20% of equity in a position
STOP_LOSS_PCT = 0.02           # 2% stop loss below entry
ATR_PERIOD = 14
ATR_THRESHOLD_PCT = 0.005      # only trade if ATR > 0.5% of price

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
    # df must have datetime index and columns: Open, High, Low, Close, Volume
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df_5min = df.resample('5min').apply(ohlc_dict).dropna()
    return df_5min

def backtest_ticker(ticker_file):
    df = pd.read_parquet(ticker_file)
    df = df.sort_values('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    # Resample to 5 minutes
    df5 = resample_to_5min(df)
    if df5.empty:
        return {'ticker': ticker_file.stem.replace('_clean',''), 'total_return':0, 'annual_return':0, 'sharpe':np.nan, 'max_drawdown':0, 'trades':0, 'win_rate':np.nan, 'avg_win':0, 'avg_loss':0, 'profit_factor':np.nan, 'final_equity':INITIAL_CAPITAL}, [], pd.DataFrame()
    # Indicators on 5‑min data
    close = df5['Close']
    high = df5['High']
    low = df5['Low']
    df5['sma_fast'] = close.rolling(window=FAST, min_periods=FAST).mean()
    df5['sma_slow'] = close.rolling(window=SLOW, min_periods=SLOW).mean()
    df5['signal'] = 0
    df5.loc[df5['sma_fast'] > df5['sma_slow'], 'signal'] = 1
    df5['position'] = df5['signal'].diff()
    # ATR
    df5['atr'] = compute_atr(df5, ATR_PERIOD)
    # Trading variables
    equity = INITIAL_CAPITAL
    position = 0          # number of shares
    entry_price = 0
    stop_loss_price = 0.0
    trades = []
    equity_curve = []
    for idx, row in df5.iterrows():
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
                # Volatility filter: require ATR > threshold
                if atr <= ATR_THRESHOLD_PCT * price:
                    continue
                if equity <= 1.0:
                    continue
                # Risk‑based sizing
                risk_amount = equity * RISK_PER_TRADE
                stop_loss = price * (1.0 - STOP_LOSS_PCT)
                if price <= stop_loss:
                    continue
                shares_risk = risk_amount / (price - stop_loss)
                # Max position size
                shares_max = (equity * MAX_POSITION_FRACTION) / price
                shares = min(shares_risk, shares_max)
                if shares <= 0:
                    continue
                # Commission calculation
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
    # Close any remaining position at end
    if position > 0:
        price = df5.iloc[-1]['Close']
        revenue = position * price
        if position * 0.005 >= 1.0:
            commission = position * 0.005
        else:
            commission = 1.0
        proceeds = revenue - commission
        equity += proceeds
        pnl = (price - entry_price) * position - commission
        trades.append({'type': 'exit_final', 'timestamp': df5.index[-1], 'price': price, 'shares': position, 'commission': commission, 'pnl': pnl})
        position = 0
        entry_price = 0
        stop_loss_price = 0.0
    equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
    stats = compute_stats(equity_curve_df['equity'], trades, ticker_file.stem.replace('_clean', ''))
    return stats, trades, equity_curve_df

def main():
    # Test on a few tickers first
    files = list(TICKERS_DIR.glob('*_clean.parquet'))[:5]
    all_stats = []
    for parquet_file in files:
        print(f'Processing {parquet_file.name} (5‑min)')
        try:
            stats, trades, equity_curve = backtest_ticker(parquet_file)
            all_stats.append(stats)
            equity_curve.to_csv(RESULTS_DIR / f"{parquet_file.stem}_5min_equity.csv")
            pd.DataFrame(trades).to_csv(RESULTS_DIR / f"{parquet_file.stem}_5min_trades.csv", index=False)
            print(f'  Trades: {len(trades)}')
            if trades:
                print(f'  First trade: {trades[0]}')
        except Exception as e:
            print(f'Error processing {parquet_file.name}: {e}')
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(RESULTS_DIR / 'summary_5min_all.csv', index=False)
        print('Saved 5‑min summary to results/summary_5min_all.csv')
    else:
        print('No stats generated')

if __name__ == '__main__':
    main()