#!/usr/bin/env python3
"""
SMA 20/50 Crossover with volatility‑scaled position sizing and regime filter:
- Entry: SMA20 > SMA50 AND (close > SMA100 OR ADX > 20)
- Stop‑loss: ATR(14) * 2.0 below entry price (volatility‑scaled risk)
- Position size: risk_per_trade = 1% of equity, max_position_frac = 10%
- Commission: $0.005/share, min $1/trade
- Slippage: fixed 0.001 per share (can be upgraded)
- Latency: next bar open
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
TICKERS_DIR = CACHE_DIR / 'scratch' / 'tickers'
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
FAST = 20
SLOW = 50
SMA_TREND = 100
ATR_PERIOD = 14
ATR_MULT = 2.0          # stop distance = ATR * ATR_MULT
ADX_PERIOD = 14
ADX_THRESH = 20
RISK_PER_TRADE = 0.01   # 1% of equity
MAX_POSITION_FRAC = 0.10 # max 10% of equity per position
SLIPPAGE_PER_SHARE = 0.001
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0

def compute_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def compute_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx

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
    # Indicators
    df['sma_fast'] = close.rolling(window=FAST, min_periods=FAST).mean()
    df['sma_slow'] = close.rolling(window=SLOW, min_periods=SLOW).mean()
    df['sma_trend'] = close.rolling(window=SMA_TREND, min_periods=SMA_TREND).mean()
    df['atr'] = compute_atr(df, ATR_PERIOD)
    df['adx'] = compute_adx(df, ADX_PERIOD)
    # Signal
    df['signal'] = 0
    df.loc[(df['sma_fast'] > df['sma_slow']) & ((close > df['sma_trend']) | (df['adx'] > ADX_THRESH)), 'signal'] = 1
    df['position'] = df['signal'].diff()
    # Latency: next bar open
    df['next_open'] = open_price.shift(-1)
    equity = INITIAL_CAPITAL
    position = 0  # number of shares
    entry_price = 0.0
    entry_atr = 0.0
    trades = []
    equity_curve = []
    for idx, row in df.iterrows():
        price_close = row['Close']
        # Execution price (next bar open) with slippage
        if pd.isna(row['next_open']):
            exec_price = price_close
        else:
            exec_price = row['next_open']
        # Equity before execution
        equity_curve.append({'timestamp': idx, 'equity': equity + position * exec_price})
        # Entry
        if row['position'] == 1 and position == 0:
            # Volatility‑scaled stop distance
            atr_val = row['atr'] if not pd.isna(row['atr']) else 0.0
            if atr_val <= 0:
                atr_val = price_close * 0.01  # fallback 1% of price
            dollar_risk_per_share = atr_val * ATR_MULT
            if dollar_risk_per_share <= 0:
                dollar_risk_per_share = price_close * 0.01
            risk_amount = equity * RISK_PER_TRADE
            shares_by_risk = risk_amount / dollar_risk_per_share
            max_shares_by_frac = (equity * MAX_POSITION_FRAC) / exec_price
            shares = min(shares_by_risk, max_shares_by_frac)
            # Slippage & commission
            price_with_slippage = exec_price + SLIPPAGE_PER_SHARE  # buy slippage
            if shares * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                commission_per_share = COMMISSION_PER_SHARE
            else:
                commission_per_share = COMMISSION_MIN_PER_TRADE / shares if shares > 0 else 0
            cost = shares * price_with_slippage + shares * commission_per_share
            if cost > equity or shares <= 0:
                continue
            position = shares
            equity -= cost
            entry_price = price_with_slippage
            entry_atr = atr_val
            trades.append({
                'type': 'entry', 'timestamp': idx, 'price': exec_price,
                'shares': shares, 'commission': shares * commission_per_share,
                'pnl': 0, 'slippage': SLIPPAGE_PER_SHARE
            })
        # Exit: signal reversal or stop‑loss
        exit_signal = (row['position'] == -1 and position > 0)
        stop_hit = False
        if position > 0:
            # Update stop price based on entry ATR (could also trail, but we use fixed stop)
            stop_price = entry_price - (entry_atr * ATR_MULT)
            if price_close <= stop_price:
                stop_hit = True
        if exit_signal or stop_hit:
            if position > 0:
                if pd.isna(row['next_open']):
                    exec_price = price_close
                else:
                    exec_price = row['next_open']
                price_with_slippage = exec_price - SLIPPAGE_PER_SHARE  # sell slippage
                if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                    commission_per_share = COMMISSION_PER_SHARE
                else:
                    commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
                proceeds = position * price_with_slippage - position * commission_per_share
                equity += proceeds
                pnl = (price_with_slippage - entry_price) * position - position * commission_per_share
                trades.append({
                    'type': 'exit', 'timestamp': idx, 'price': exec_price,
                    'shares': position, 'commission': position * commission_per_share,
                    'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE
                })
                position = 0
                entry_price = 0.0
                entry_atr = 0.0
    # Close any remaining position at end
    if position > 0:
        price_close = df.iloc[-1]['Close']
        exec_price = price_close  # no next bar
        price_with_slippage = exec_price - SLIPPAGE_PER_SHARE
        if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
            commission_per_share = COMMISSION_PER_SHARE
        else:
            commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
        proceeds = position * price_with_slippage - position * commission_per_share
        equity += proceeds
        pnl = (price_with_slippage - entry_price) * position - position * commission_per_share
        trades.append({
            'type': 'exit', 'timestamp': df.index[-1], 'price': exec_price,
            'shares': position, 'commission': position * commission_per_share,
            'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE
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
            equity_curve.to_csv(RESULTS_DIR / f"{parquet_file.stem}_equity_voladj.csv")
            pd.DataFrame(trades).to_csv(RESULTS_DIR / f"{parquet_file.stem}_trades_voladj.csv", index=False)
        except Exception as e:
            print(f'Error processing {parquet_file.name}: {e}')
            import traceback
            traceback.print_exc()
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(RESULTS_DIR / 'summary_voladj_all.csv', index=False)
        print('Saved summary to results/summary_voladj_all.csv')
        print('\n=== Aggregate Results (Vol‑Adj + Regime Filter) ===')
        print(f'Mean total return: {stats_df["total_return"].mean():.2%}')
        print(f'Median total return: {stats_df["total_return"].median():.2%}')
        print(f'Mean Sharpe: {stats_df["sharpe"].mean():.2f}')
        print(f'Mean max drawdown: {stats_df["max_drawdown"].mean():.2%}')
    else:
        print('No stats generated')

if __name__ == '__main__':
    main()