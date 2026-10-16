#!/usr/bin/env python3
"""
SMA 20/50 crossover with ML signal (logistic regression) using a simple train/test split.
Features: recent returns, volatility, volume, RSI, price vs SMAs.
Trains on first 80% of data, tests on last 20%.
Outputs per-ticker statistics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, will use dummy model (always predict 0.5)")

CACHE_DIR = Path(os.getenv('HERMES_CACHE', '/home/hermes/.hermes/cache'))
TICKERS_DIR = CACHE_DIR / 'scratch' / 'tickers'
RESULTS_DIR = CACHE_DIR / 'scratch' / 'results_ml_simple'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
FAST = 20
SLOW = 50
SMA_TREND = 100
ATR_PERIOD = 14
ATR_MULT = 2.0
ADX_PERIOD = 14
ADX_THRESH = 20
RISK_PER_TRADE = 0.01
MAX_POSITION_FRAC = 0.10
SLIPPAGE_PER_SHARE = 0.001
COMMISSION_PER_SHARE = 0.005
COMMISSION_MIN_PER_TRADE = 1.0
INITIAL_CAPITAL = 10000.0
TRAIN_FRAC = 0.8  # first 80% for training
PROB_THRESHOLD = 0.55  # only go long if predicted prob > this

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

def compute_rsi(df, period=14):
    close = df['Close']
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features(df):
    df = df.copy()
    close = df['Close']
    volume = df['Volume']
    # Returns
    df['ret_1'] = close.pct_change(1)
    df['ret_5'] = close.pct_change(5)
    df['ret_10'] = close.pct_change(10)
    df['ret_20'] = close.pct_change(20)
    # Volatility (std of returns)
    df['vol_5'] = df['ret_1'].rolling(window=5).std()
    df['vol_10'] = df['ret_1'].rolling(window=10).std()
    # Volume change
    df['vol_chg'] = volume.pct_change(1)
    df['vol_ma_ratio'] = volume / volume.rolling(window=10).mean()
    # RSI
    df['rsi'] = compute_rsi(df, 14)
    return df

def compute_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    df['sma_fast'] = close.rolling(window=FAST, min_periods=FAST).mean()
    df['sma_slow'] = close.rolling(window=SLOW, min_periods=SLOW).mean()
    df['sma_trend'] = close.rolling(window=SMA_TREND, min_periods=SMA_TREND).mean()
    df['atr'] = compute_atr(df, ATR_PERIOD)
    df['adx'] = compute_adx(df, ADX_PERIOD)
    df['rsi'] = compute_rsi(df, 14)
    return df

def compute_stats(equity_curve, trades):
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
        'ticker': None,
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

def backtest_ticker(ticker_file):
    df = pd.read_parquet(ticker_file)
    df = df.sort_values('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    # Compute indicators
    df = compute_indicators(df)
    # Compute features
    df = compute_features(df)
    # Target: next bar up (close.shift(-1) > close)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    # Drop rows where target is NaN (last bar)
    df = df.dropna(subset=['target'])
    # Feature columns
    feature_cols = ['ret_1', 'ret_5', 'ret_10', 'ret_20',
                    'vol_5', 'vol_10', 'vol_chg', 'vol_ma_ratio',
                    'rsi',
                    'sma_fast', 'sma_slow', 'sma_trend', 'atr', 'adx']
    df[feature_cols] = df[feature_cols].fillna(0)
    # Split
    split_idx = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    if len(train) == 0 or len(test) == 0:
        print(f'  Not enough data for split')
        return None, None, None
    # Train model
    X_train = train[feature_cols]
    y_train = train['target']
    if SKLEARN_AVAILABLE:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, n_jobs=1))
        model.fit(X_train, y_train)
        probs = model.predict_proba(test[feature_cols])[:, 1]
    else:
        probs = np.full(len(test), 0.5)
    # Backtest on test period
    equity = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    trades = []
    equity_curve = []
    # We need to iterate over test rows with index to align probabilities
    test_indices = test.index
    for i, idx in enumerate(test_indices):
        row = test.loc[idx]
        price_close = row['Close']
        # Execution price: next bar open with slippage
        # Determine next open: if next row exists in original df, use its Open, else close
        loc_idx = df.index.get_loc(idx)
        if loc_idx + 1 < len(df):
            next_open = df.iloc[loc_idx + 1]['Open']
        else:
            next_open = price_close
        price_mid = next_open  # for equity calculation
        # Update equity curve before execution
        equity_curve.append({'timestamp': idx, 'equity': equity + position * next_open})
        # Entry signal
        signal = 1 if row['sma_fast'] > row['sma_slow'] else 0
        if signal == 1 and position == 0 and probs[i] > PROB_THRESHOLD:
            atr_val = row['atr'] if not pd.isna(row['atr']) else 0.0
            if atr_val <= 0:
                atr_val = price_close * 0.01
            dollar_risk_per_share = atr_val * ATR_MULT
            if dollar_risk_per_share <= 0:
                dollar_risk_per_share = price_close * 0.01
            risk_amount = equity * RISK_PER_TRADE
            shares_by_risk = risk_amount / dollar_risk_per_share
            max_shares_by_frac = (equity * MAX_POSITION_FRAC) / next_open
            shares = min(shares_by_risk, max_shares_by_frac)
            price_with_slippage = next_open + SLIPPAGE_PER_SHARE  # buy slippage
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
                'type': 'entry', 'timestamp': idx, 'price': next_open,
                'shares': shares, 'commission': shares * commission_per_share,
                'pnl': 0, 'slippage': SLIPPAGE_PER_SHARE
            })
        # Exit: signal reversal or stop‑loss
        exit_signal = 0
        if position > 0:
            signal = 1 if row['sma_fast'] > row['sma_slow'] else 0
            if signal == 0:  # crossover down
                exit_signal = 1
            # Stop‑loss
            stop_price = entry_price - (entry_atr * ATR_MULT)
            if price_close <= stop_price:
                exit_signal = 1
        if exit_signal and position > 0:
            loc_idx = df.index.get_loc(idx)
            if loc_idx + 1 < len(df):
                next_open_exit = df.iloc[loc_idx + 1]['Open']
            else:
                next_open_exit = price_close
            price_with_slippage = next_open_exit - SLIPPAGE_PER_SHARE  # sell slippage
            if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
                commission_per_share = COMMISSION_PER_SHARE
            else:
                commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
            proceeds = position * price_with_slippage - position * commission_per_share
            equity += proceeds
            pnl = (price_with_slippage - entry_price) * position - position * commission_per_share
            trades.append({
                'type': 'exit', 'timestamp': idx, 'price': next_open_exit,
                'shares': position, 'commission': position * commission_per_share,
                'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE
            })
            position = 0
            entry_price = 0.0
            entry_atr = 0.0
    # Close any remaining position at end using close price (no next bar)
    if position > 0:
        price_close = test.iloc[-1]['Close']
        price_with_slippage = price_close - SLIPPAGE_PER_SHARE
        if position * COMMISSION_PER_SHARE >= COMMISSION_MIN_PER_TRADE:
            commission_per_share = COMMISSION_PER_SHARE
        else:
            commission_per_share = COMMISSION_MIN_PER_TRADE / position if position > 0 else 0
        proceeds = position * price_with_slippage - position * commission_per_share
        equity += proceeds
        pnl = (price_with_slippage - entry_price) * position - position * commission_per_share
        trades.append({
            'type': 'exit', 'timestamp': test.index[-1], 'price': price_close,
            'shares': position, 'commission': position * commission_per_share,
            'pnl': pnl, 'slippage': SLIPPAGE_PER_SHARE
        })
        position = 0
    equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
    stats = compute_stats(equity_df['equity'], trades)
    stats['ticker'] = ticker_file.stem.replace('_clean', '')
    return stats, trades, equity_df

def main():
    all_stats = []
    # Limit to first 5 tickers for speed
    ticker_files = list(TICKERS_DIR.glob('*_clean.parquet'))[:5]
    for parquet_file in ticker_files:
        print(f'Processing {parquet_file.name}')
        try:
            stats, trades, equity_curve = backtest_ticker(parquet_file)
            if stats is None:
                continue
            all_stats.append(stats)
            equity_curve.to_csv(RESULTS_DIR / f"{parquet_file.stem}_equity_ml.csv")
            pd.DataFrame(trades).to_csv(RESULTS_DIR / f"{parquet_file.stem}_trades_ml.csv", index=False)
            print(f'  Sharpe: {stats.get("sharpe", "NaN")}, Return: {stats.get("total_return", 0):.2%}, Trades: {stats.get("trades", 0)}')
        except Exception as e:
            print(f'  Error: {e}')
            import traceback
            traceback.print_exc()
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(RESULTS_DIR / 'summary_ml_all.csv', index=False)
        print('\n=== Aggregate Results (ML‑filtered SMA 20/50, train/test split) ===')
        print(f'Mean total return: {stats_df["total_return"].mean():.2%}')
        print(f'Median total return: {stats_df["total_return"].median():.2%}')
        print(f'Mean Sharpe: {stats_df["sharpe"].mean():.2f}')
        print(f'Mean max drawdown: {stats_df["max_drawdown"].mean():.2%}')
    else:
        print('No stats generated')

if __name__ == '__main__':
    main()