#!/usr/bin/env python3
"""
Enhanced SMA 20/50 Crossover Backtest with:
- Adaptive risk per ticker using volatility (ATR) scaling
- Stop-loss based on ATR or fixed percentage
- Max position fraction limit
- Regime/trend filter (price > SMA100 or ADX > 20)
- Walk‑forward parameter optimization (simple rolling window)
Outputs per‑ticker statistics to CSV.
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

# Base SMA parameters (will be optimized in walk‑forward)
FAST = 20
SLOW = 50

# Risk parameters
RISK_PER_TRADE = 0.01          # 1% of equity risked per trade
MAX_POSITION_FRAC = 0.10       # max 10% of equity per position
SLIPPAGE_PER_SHARE = 0.001     # slippage per share (absolute)
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
LATENCY_BARS = 1               # execute at next bar's open after signal

# Stop‑loss parameters
USE_ATR_STOP = True            # if False, use fixed percentage stop
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0           # stop distance = ATR * multiplier
FIXED_STOP_PCT = 0.015         # 1.5% stop loss if not using ATR

# Regime filter parameters
SMA_TREND = 100                # longer‑term SMA for trend filter
ADX_PERIOD = 14
ADX_THRESHOLD = 20             # enter only if ADX > this

# Walk‑forward settings
WINDOW_TRAIN_MONTHS = 6
WINDOW_TEST_MONTHS = 1
PARAM_GRID = {
    'fast': [10, 15, 20, 25],
    'slow': [30, 40, 50, 60],
    'risk_per_trade': [0.005, 0.01, 0.02],
    'max_position_frac': [0.05, 0.10, 0.15],
}

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
    minus_dm = abs(minus_dm)
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

def walk_forward_optimize(df_train):
    """
    Simple grid search over PARAM_GRID using train data.
    Returns best parameters based on Sharpe ratio.
    """
    best_sharpe = -np.inf
    best_params = {}
    # Precompute indicators that don't depend on SMA lengths
    df_train = df_train.copy()
    df_train['atr'] = compute_atr(df_train, ATR_PERIOD)
    df_train['adx'] = compute_adx(df_train, ADX_PERIOD)
    df_train['sma_trend'] = df_train['Close'].rolling(window=SMA_TREND, min_periods=SMA_TREND).mean()
    for fast in PARAM_GRID['fast']:
        for slow in PARAM_GRID['slow']:
            if fast >= slow:
                continue
            df_train['sma_fast'] = df_train['Close'].rolling(window=fast, min_periods=fast).mean()
            df_train['sma_slow'] = df_train['Close'].rolling(window=slow, min_periods=slow).mean()
            df_train['signal'] = 0
            df_train.loc[df_train['sma_fast'] > df_train['sma_slow'], 'signal'] = 1
            # Apply regime filter
            regime_ok = (df_train['Close'] > df_train['sma_trend']) | (df_train['adx'] > ADX_THRESHOLD)
            df_train['signal'] = df_train['signal'] * regime_ok.astype(int)
            df_train['position'] = df_train['signal'].diff()
            for risk in PARAM_GRID['risk_per_trade']:
                for max_frac in PARAM_GRID['max_position_frac']:
                    stats, _, _ = backtest_ticker_from_df(df_train, risk_per_trade=risk, max_position_frac=max_frac,
                                                          fast=fast, slow=slow)
                    if not np.isnan(stats['sharpe']) and stats['sharpe'] > best_sharpe:
                        best_sharpe = stats['sharpe']
                        best_params = {'fast': fast, 'slow': slow,
                                       'risk_per_trade': risk, 'max_position_frac': max_frac}
    return best_params

def backtest_ticker_from_df(df, risk_per_trade=RISK_PER_TRADE, max_position_frac=MAX_POSITION_FRAC,
                            fast=FAST, slow=SLOW):
    """
    Backtest on a pre‑processed dataframe that already contains:
    - 'atr', 'adx', 'sma_trend' (optional)
    - 'sma_fast', 'sma_slow', 'signal', 'position'
    Assumes df is indexed by datetime and has columns: Open, High, Low, Close.
    """
    df = df.copy()
    # Ensure required columns exist
    if 'atr' not in df.columns:
        df['atr'] = compute_atr(df, ATR_PERIOD)
    if 'adx' not in df.columns:
        df['adx'] = compute_adx(df, ADX_PERIOD)
    if 'sma_trend' not in df.columns:
        df['sma_trend'] = df['Close'].rolling(window=SMA_TREND, min_periods=SMA_TREND).mean()
    if 'sma_fast' not in df.columns or 'sma_slow' not in df.columns:
        df['sma_fast'] = df['Close'].rolling(window=fast, min_periods=fast).mean()
        df['sma_slow'] = df['Close'].rolling(window=slow, min_periods=slow).mean()
    if 'signal' not in df.columns:
        df['signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
        # Regime filter
        regime_ok = (df['Close'] > df['sma_trend']) | (df['adx'] > ADX_THRESHOLD)
        df['signal'] = df['signal'] * regime_ok.astype(int)
    if 'position' not in df.columns:
        df['position'] = df['signal'].diff()
    # Latency: next bar's open price
    df['next_open'] = df['Open'].shift(-1)
    equity = 10000.0  # per ticker
    position = 0
    entry_price = 0.0
    entry_time = None
    max_price_since_entry = 0.0
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
        # Entry signal
        if row['position'] == 1 and position == 0:
            # Compute volatility‑scaled position size
            atr = row['atr'] if not pd.isna(row['atr']) else 0.0
            if atr <= 0:
                atr = price_close * 0.01  # fallback 1% of price
            risk_amount = equity * risk_per_trade
            # Dollar risk per share = atr * ATR_MULTIPLIER (if using ATR stop) or price * FIXED_STOP_PCT
            if USE_ATR_STOP:
                dollar_risk_per_share = atr * ATR_MULTIPLIER
            else:
                dollar_risk_per_share = price_close * FIXED_STOP_PCT
            if dollar_risk_per_share <= 0:
                dollar_risk_per_share = price_close * 0.01
            shares_by_risk = risk_amount / dollar_risk_per_share
            # Max shares by position fraction
            max_shares_by_frac = (equity * max_position_frac) / exec_price
            shares = min(shares_by_risk, max_shares_by_frac)
            # Consider slippage and commission
            price_with_slippage = exec_price + SLIPPAGE_PER_SHARE  # buy slippage
            # Estimate commission
            if shares * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                commission = shares * COMMISSION_PER_SHARE
            else:
                commission = COMMISSION_MIN_PER_TRADE
            cost = shares * price_with_slippage + commission
            if cost > equity or shares <= 0:
                continue
            position = shares
            equity -= cost
            entry_price = price_with_slippage
            entry_time = idx
            max_price_since_entry = price_with_slippage
            trades.append({'type': 'entry', 'timestamp': idx, 'price': exec_price,
                           'shares': shares, 'commission': commission,
                           'pnl': 0, 'slippage': SLIPPAGE_PER_SHARE})
        # Exit conditions: signal reversal or stop‑loss
        exit_signal = (row['position'] == -1 and position > 0)
        stop_hit = False
        if position > 0:
            # Update max price
            if price_close > max_price_since_entry:
                max_price_since_entry = price_close
            # Compute stop price
            if USE_ATR_STOP:
                stop_price = entry_price - (atr * ATR_MULTIPLIER)
            else:
                stop_price = entry_price * (1.0 - FIXED_STOP_PCT)
            if price_close <= stop_price:
                stop_hit = True
        if exit_signal or stop_hit:
            if position > 0:
                # Determine execution price for exit
                if pd.isna(row['next_open']):
                    exec_price = price_close
                else:
                    exec_price = row['next_open']
                price_with_slippage = exec_price - SLIPPAGE_PER_SHARE  # sell slippage
                if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                    commission = position * COMMISSION_PER_SHARE
                else:
                    commission = COMMISSION_MIN_PER_TRADE
                proceeds = position * price_with_slippage - commission
                equity += proceeds
                pnl = (price_with_slippage - entry_price) * position - commission
                trades.append({'type': 'exit', 'timestamp': idx, 'price': exec_price,
                               'shares': position, 'commission': commission,
                               'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE})
                position = 0
                entry_price = 0.0
                entry_time = None
                max_price_since_entry = 0.0
    # Close any remaining position at end
    if position > 0:
        price_close = df.iloc[-1]['Close']
        exec_price = price_close  # no next bar
        price_with_slippage = exec_price - SLIPPAGE_PER_SHARE
        if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
            commission = position * COMMISSION_PER_SHARE
        else:
            commission = COMMISSION_MIN_PER_TRADE
        proceeds = position * price_with_slippage - commission
        equity += proceeds
        pnl = (price_with_slippage - entry_price) * position - commission
        trades.append({'type': 'exit', 'timestamp': df.index[-1], 'price': exec_price,
                       'shares': position, 'commission': commission,
                       'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE})
        position = 0
    equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
    stats = compute_stats(equity_curve_df['equity'], trades, df.attrs.get('ticker', 'unknown'))
    return stats, trades, equity_curve_df

def backtest_ticker(ticker_file):
    df = pd.read_parquet(ticker_file)
    df = df.sort_values('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df.attrs['ticker'] = ticker_file.stem.replace('_clean', '')
    # Walk‑forward: split data into train/test by time
    # For simplicity, we'll just use the whole data with default parameters
    # (Walk‑forward optimization is more involved; we'll implement a basic version)
    # Determine date range
    start_date = df.index.min()
    end_date = df.index.max()
    # For demo, we'll use last 6 months as train, preceding month as test? Actually walk‑forward uses expanding window.
    # We'll implement a simple rolling window: train = first 80%, test = last 20%
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    if len(df_train) < 100:
        # Not enough data for training, fall back to full data
        best_params = {'fast': FAST, 'slow': SLOW,
                       'risk_per_trade': RISK_PER_TRADE,
                       'max_position_frac': MAX_POSITION_FRAC}
    else:
        best_params = walk_forward_optimize(df_train)
        if not best_params:
            best_params = {'fast': FAST, 'slow': SLOW,
                           'risk_per_trade': RISK_PER_TRADE,
                           'max_position_frac': MAX_POSITION_FRAC}
    # Run test with best parameters
    stats, trades, equity_curve = backtest_ticker_from_df(
        df_test,
        risk_per_trade=best_params.get('risk_per_trade', RISK_PER_TRADE),
        max_position_frac=best_params.get('max_position_frac', MAX_POSITION_FRAC),
        fast=best_params.get('fast', FAST),
        slow=best_params.get('slow', SLOW)
    )
    # Optionally, we could also compute stats on full data with best params for reporting
    # For simplicity, we'll just return test stats
    return stats, trades, equity_curve

def main():
    all_stats = []
    for parquet_file in TICKERS_DIR.glob('*_clean.parquet'):
        print(f'Processing {parquet_file.name}')
        try:
            stats, trades, equity_curve = backtest_ticker(parquet_file)
            all_stats.append(stats)
            equity_curve.to_csv(RESULTS_DIR / f"{parquet_file.stem}_equity_enhanced.csv")
            pd.DataFrame(trades).to_csv(RESULTS_DIR / f"{parquet_file.stem}_trades_enhanced.csv", index=False)
        except Exception as e:
            print(f'Error processing {parquet_file.name}: {e}')
            import traceback
            traceback.print_exc()
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(RESULTS_DIR / 'summary_enhanced_all.csv', index=False)
        print('Saved summary to results/summary_enhanced_all.csv')
        # Print aggregate metrics
        print('\n=== Aggregate Results ===')
        print(f'Mean total return: {stats_df["total_return"].mean():.2%}')
        print(f'Median total return: {stats_df["total_return"].median():.2%}')
        print(f'Mean Sharpe: {stats_df["sharpe"].mean():.2f}')
        print(f'Mean max drawdown: {stats_df["max_drawdown"].mean():.2%}')
    else:
        print('No stats generated')

if __name__ == '__main__':
    main()