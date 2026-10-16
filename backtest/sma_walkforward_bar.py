#!/usr/bin/env python3
"""
Walk‑forward optimization of SMA crossover with ATR‑based stop loss.
Bar size configurable (1,5,15 minutes).
Parameters optimized: fast SMA, slow SMA, risk per trade, ATR multiplier.
Uses expanding window: train = first 6 months, test = next month, roll forward 1 month.
Outputs per‑ticker test performance and aggregate statistics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import itertools

# -------------------- Configuration --------------------
CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
TICKERS_DIR = CACHE_DIR / 'scratch' / 'tickers'
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results_wf'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Bar sizes to test (in minutes)
BAR_SIZES = [1, 5, 15]          # will process each separately
# Parameter grids
FAST_GRID = [10, 15, 20, 25]
SLOW_GRID = [30, 40, 50, 60]   # ensure slow > fast
RISK_GRID = [0.005, 0.01, 0.02]   # fraction of equity risked per trade
ATR_MULT_GRID = [1.5, 2.0, 2.5]   # stop distance = ATR * multiplier
# Fixed parameters
MAX_POSITION_FRAC = 0.10        # max 10% of equity per position
SLIPPAGE_PER_SHARE = 0.001      # slippage per share (absolute)
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0       # per ticker for simplicity
TRAIN_MONTHS = 6
TEST_MONTHS = 1
# -------------------------------------------------------

def resample_ohlcv(df, bar_size_min):
    """Resample 1‑minute OHLCV to given bar size in minutes."""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    freq = f'{bar_size_min}min'
    # Resample
    resampled = df.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()  # drop incomplete periods
    return resampled

def compute_indicators(df, fast, slow):
    """Add SMA fast/slow and ATR(14) columns to df."""
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    df['sma_fast'] = close.rolling(window=fast, min_periods=fast).mean()
    df['sma_slow'] = close.rolling(window=slow, min_periods=slow).mean()
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14, min_periods=14).mean()
    return df

def compute_stats(equity_curve, trades):
    """Return dict of performance metrics."""
    if len(equity_curve) < 2:
        return {'total_return': np.nan, 'sharpe': np.nan, 'max_drawdown': np.nan, 'trades': 0}
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
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_equity': equity_curve.iloc[-1]
    }

def backtest_ticker(df, fast, slow, risk, atr_mult):
    """Backtest on already resampled and indicator‑rich dataframe.
    Returns stats dict, trades list, equity curve series."""
    df = df.copy()
    # Ensure indicators exist
    if 'sma_fast' not in df.columns:
        df = compute_indicators(df, fast, slow)
    # Signal
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df['position'] = df['signal'].diff()
    # Latency: execute at next bar's open
    df['next_open'] = df['Open'].shift(-1)
    equity = INITIAL_CAPITAL
    position = 0  # number of shares
    entry_price = 0.0
    entry_atr = 0.0
    trades = []
    equity_curve = []
    for idx, row in df.iterrows():
        # Price before execution (use next_open if available, else close)
        price_mid = row['next_open'] if not pd.isna(row['next_open']) else row['Close']
        equity_curve.append({'timestamp': idx, 'equity': equity + position * price_mid})
        # Entry
        if row['position'] == 1 and position == 0:
            atr_val = row['atr'] if not pd.isna(row['atr']) else 0.0
            if atr_val <= 0:
                atr_val = row['Close'] * 0.01  # fallback 1% of price
            dollar_risk_per_share = atr_val * atr_mult
            if dollar_risk_per_share <= 0:
                dollar_risk_per_share = row['Close'] * 0.01
            risk_amount = equity * risk
            shares_by_risk = risk_amount / dollar_risk_per_share
            max_shares_by_frac = (equity * MAX_POSITION_FRAC) / (row['next_open'] if not pd.isna(row['next_open']) else row['Close'])
            shares = min(shares_by_risk, max_shares_by_frac)
            # Slippage & commission
            price_with_slippage = (row['next_open'] + SLIPPAGE_PER_SHARE) if not pd.isna(row['next_open']) else (row['Close'] + SLIPPAGE_PER_SHARE)
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
                'type': 'entry', 'timestamp': idx, 'price': row['next_open'] if not pd.isna(row['next_open']) else row['Close'],
                'shares': shares, 'commission': shares * commission_per_share,
                'pnl': 0, 'slippage': SLIPPAGE_PER_SHARE
            })
        # Exit: signal reversal or stop‑loss
        exit_signal = (row['position'] == -1 and position > 0)
        stop_hit = False
        if position > 0:
            stop_price = entry_price - (entry_atr * atr_mult)
            if row['Close'] <= stop_price:
                stop_hit = True
        if exit_signal or stop_hit:
            if position > 0:
                price_exec = row['next_open'] if not pd.isna(row['next_open']) else row['Close']
                price_with_slippage = price_exec - SLIPPAGE_PER_SHARE  # sell slippage
                if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                    commission_per_share = COMMISSION_PER_SHARE
                else:
                    commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
                proceeds = position * price_with_slippage - position * commission_per_share
                equity += proceeds
                pnl = (price_with_slippage - entry_price) * position - position * commission_per_share
                trades.append({
                    'type': 'exit', 'timestamp': idx, 'price': price_exec,
                    'shares': position, 'commission': position * commission_per_share,
                    'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE
                })
                position = 0
                entry_price = 0.0
                entry_atr = 0.0
    # Close any remaining position at end using close price (no next bar)
    if position > 0:
        price_close = df.iloc[-1]['Close']
        price_with_slippage = price_close - SLIPPAGE_PER_SHARE
        if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
            commission_per_share = COMMISSION_PER_SHARE
        else:
            commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
        proceeds = position * price_with_slippage - position * commission_per_share
        equity += proceeds
        pnl = (price_with_slippage - entry_price) * position - position * commission_per_share
        trades.append({
            'type': 'exit', 'timestamp': df.index[-1], 'price': price_close,
            'shares': position, 'commission': position * commission_per_share,
            'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE
        })
        position = 0
    equity_curve_df = pd.DataFrame(equity_curve).set_index('timestamp')
    stats = compute_stats(equity_curve_df['equity'], trades)
    return stats, trades, equity_curve_df['equity']

def walk_forward_optimize(df_resampled):
    """Perform expanding window walk‑forward optimization.
    Returns list of test‑period stats dicts (one per test window)."""
    # Ensure index is datetime
    if not isinstance(df_resampled.index, pd.DatetimeIndex):
        df_resampled = df_resampled.copy()
        df_resampled.index = pd.to_datetime(df_resampled.index)
    # Determine start and end
    start_date = df_resampled.index.min()
    end_date = df_resampled.index.max()
    # Compute month offsets for window start
    from pandas.tseries.offsets import MonthEnd
    train_offset = MonthEnd(TRAIN_MONTHS)
    test_offset = MonthEnd(TEST_MONTHS)
    # First train window end
    train_end = start_date + train_offset
    results = []
    while True:
        test_start = train_end + pd.Timedelta(days=1)  # next bar after train end
        test_end = test_start + test_offset
        if test_start > end_date:
            break
        # Slice data
        train_data = df_resampled.loc[start_date:train_end]
        test_data = df_resampled.loc[test_start:test_end]
        if len(train_data) < 30 or len(test_data) < 10:
            # Not enough data, shift window forward by one test month and continue
            train_end = test_end
            continue
        # Grid search on train data
        best_sharpe = -np.inf
        best_params = None
        for fast, slow, risk, atr_mult in itertools.product(FAST_GRID, SLOW_GRID, RISK_GRID, ATR_MULT_GRID):
            if fast >= slow:
                continue
            stats, _, _ = backtest_ticker(train_data.copy(), fast, slow, risk, atr_mult)
            if not np.isnan(stats['sharpe']) and stats['sharpe'] > best_sharpe:
                best_sharpe = stats['sharpe']
                best_params = (fast, slow, risk, atr_mult)
        if best_params is None:
            # fallback to default
            best_params = (FAST_GRID[0], SLOW_GRID[0], RISK_GRID[0], ATR_MULT_GRID[0])
        # Evaluate on test data with best params
        fast, slow, risk, atr_mult = best_params
        test_stats, _, _ = backtest_ticker(test_data.copy(), fast, slow, risk, atr_mult)
        results.append({
            'train_start': start_date,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'params': best_params,
            'stats': test_stats
        })
        # Roll window forward by test period
        train_end = test_end
    return results

def process_ticker(ticker_file, bar_size):
    df_1min = pd.read_parquet(ticker_file)
    df_1min = df_1min.sort_values('datetime')
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])
    df_1min = df_1min.set_index('datetime')
    df_resampled = resample_ohlcv(df_1min, bar_size)
    if df_resampled.empty:
        return []
    wf_results = walk_forward_optimize(df_resampled)
    # Attach ticker name
    ticker = ticker_file.stem.replace('_clean', '')
    for r in wf_results:
        r['ticker'] = ticker
        r['bar_size'] = bar_size
    return wf_results

def main():
    all_results = []
    # Limit to first 5 tickers for speed; comment out to run all
    ticker_files = list(TICKERS_DIR.glob('*_clean.parquet'))[:5]
    for bar_size in BAR_SIZES:
        print(f'\n=== Processing bar size {bar_size} minute ===')
        for ticker_file in ticker_files:
            print(f'  Processing {ticker_file.name}...')
            try:
                results = process_ticker(ticker_file, bar_size)
                all_results.extend(results)
                print(f'    -> {len(results)} windows')
            except Exception as e:
                print(f'    Error: {e}')
                import traceback
                traceback.print_exc()
    # Convert to flat list of rows for CSV
    rows = []
    for r in all_results:
        base = {
            'ticker': r['ticker'],
            'bar_size': r['bar_size'],
            'train_start': r['train_start'],
            'train_end': r['train_end'],
            'test_start': r['test_start'],
            'test_end': r['test_end'],
            'fast': r['params'][0],
            'slow': r['params'][1],
            'risk': r['params'][2],
            'atr_mult': r['params'][3]
        }
        # Flatten stats
        for k, v in r['stats'].items():
            base[f'stat_{k}'] = v
        rows.append(base)
    results_df = pd.DataFrame(rows)
    # Save detailed results
    detailed_path = RESULTS_DIR / 'walkforward_detailed.csv'
    results_df.to_csv(detailed_path, index=False)
    print(f'\nSaved detailed results to {detailed_path}')
    # Aggregate by bar size
    agg_rows = []
    for bs in BAR_SIZES:
        sub = results_df[results_df['bar_size'] == bs]
        if sub.empty:
            continue
        agg = {
            'bar_size': bs,
            'mean_total_return': sub['stat_total_return'].mean(),
            'median_total_return': sub['stat_total_return'].median(),
            'mean_sharpe': sub['stat_sharpe'].mean(),
            'median_sharpe': sub['stat_sharpe'].median(),
            'mean_max_drawdown': sub['stat_max_drawdown'].mean(),
            'mean_trades': sub['stat_trades'].mean(),
            'windows': len(sub)
        }
        agg_rows.append(agg)
    agg_df = pd.DataFrame(agg_rows)
    agg_path = RESULTS_DIR / 'walkforward_aggregate.csv'
    agg_df.to_csv(agg_path, index=False)
    print(f'Saved aggregate results to {agg_path}')
    print('\nAggregate results:')
    print(agg_df.to_string(index=False))

if __name__ == '__main__':
    main()