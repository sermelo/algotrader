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
        print(f\"Warning: {ticker_file.name} generated zero trades.\")
    return stats, trades, equity_curve_df