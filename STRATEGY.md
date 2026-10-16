# Strategy specification

This describes what `algotrader/` actually implements (verified by the unit tests).
The earlier version of this file contained performance claims that could not be
reproduced; they have been removed.

## Signal

Long-only SMA crossover on bar closes of any bar size (set by `--timeframe`):

- `want_long` is True while `SMA(fast) > SMA(slow)` (defaults 20 and 50 bars).
- Optional trend filter: also require `close > SMA(trend_filter)`.
- During the warm-up (until the slowest indicator exists) `want_long` is False.
- Parameters must satisfy `0 < fast < slow`, `atr_period >= 1`, `atr_mult > 0`,
  `trend_filter >= 1` (or none); otherwise construction fails.
- ATR is the simple mean of the true range over `atr_period` bars (not Wilder's).

## Decision-tree signal (`--strategy tree`)

After arXiv:2405.13959 (see `R&D/`). One model per ticker:

- **Training:** fit on bars strictly before `--train-end`; backtest only bars from
  `--train-end` on. The model never sees test bars.
- **Label:** 1 if the next bar's close is above this bar's close, else 0 (ties are 0).
- **Features** (all from data up to the bar's close): 1-bar return, 15-bar return,
  RSI(14) and ADX(14) with Wilder smoothing, SMA(14)/close, rolling 14-bar correlation
  of close with SMA(14), 14-bar volatility of 1-bar returns, 14-bar volatility of
  15-bar returns, and 14-bar VWAP (typical price (H+L+C)/3) / close. The paper gives no
  window for the correlation or VWAP; 14 is our choice. RSI is 50 when there was no
  price change at all.
- **Model:** scikit-learn `DecisionTreeClassifier`, Gini, `max_depth` 4 (`--max-depth`),
  `random_state` 0.
- `want_long` is the predicted class. Bars with any undefined feature (warm-up, zero
  volume, constant prices) are flat.
- Orders, stops, sizing and costs are exactly as below. Fresh-signal entry means the
  position opens when the prediction turns from 0 to 1 and closes at the next open
  after it turns back to 0.
- Differences from the paper: the paper used no costs and no stop, close-to-close
  returns and linear interpolation of missing bars (which uses later prices). We fill
  at the next open, drop empty bars and charge costs. The paper's train/test years
  were 2022–23 / 2023–24 on NIFTY50; ours are Mar 2022–Dec 2023 / Jan 2024 on.

## Bars

Intraday bar sizes are built from 1-minute bars anchored to the 09:30 session open
(`1h` = 09:30 to 10:30, ...; the 15:30 bar is a half bar). Daily and longer bars use
calendar periods. Periods with no trades are dropped. Indicators run straight across
the overnight gap, and positions can be held overnight.

## Orders and execution

| Event | Rule |
|---|---|
| Entry signal | `want_long` turns True (fresh crossover) while flat. |
| Entry fill | Next bar's open, plus slippage per share. |
| Exit signal | `want_long` turns False while in a position. |
| Exit fill | Next bar's open, minus slippage per share. |
| Stop price | Entry fill minus `atr_mult * ATR(atr_period)`; ATR is from the signal bar. Defaults 2.0 and 14 bars. |
| Stop trigger | The bar's low is at or below the stop. Fill at the stop, or at the open if it gaps below. |
| After a stop | No new entry until the next fresh crossover. |
| End of data | Open position closed at the last close. |

## Position size

`shares = floor(min(equity * risk / stop_distance, equity * max_frac / price))`

with `risk = 1%`, `max_frac = 10%` by default, `stop_distance = atr_mult * ATR`.
If the order plus commission exceeds cash, size is reduced until it fits.

## Costs

- Commission: $0.005 per share, $1.00 minimum per order (Interactive Brokers fixed).
- Slippage: fixed dollars per share per side, default 0 (`--slippage`).

## Metrics

Total return, CAGR, max drawdown (bar-level equity), Sharpe (daily equity returns,
sqrt(252), risk-free 0), exposure (fraction of bars ending with a position open),
trades, win rate, average win/loss (break-even trades count as losses), profit
factor (gross profit / gross loss; inf if there are wins and no losses), total
commission, buy-and-hold return (first open to last close).

## Not implemented (ideas)

Shorting, flat-at-close (intraday only) mode, ADX regime filter, volume-based partial fills, random latency,
multi-instrument portfolio, walk-forward parameter selection (and walk-forward
retraining of the decision tree). 
