# Strategy specification

This describes what `algotrader/` actually implements (verified by the unit tests).
The earlier version of this file contained performance claims that could not be
reproduced; they have been removed.

## Signal

Long-only SMA crossover on bar closes of any bar size (set by `--timeframe`):

- `want_long` is True while `SMA(fast) > SMA(slow)` (defaults 20 and 50 bars).
- Optional trend filter: also require `close > SMA(trend_filter)`.
- During the warm-up (until the slowest indicator exists) `want_long` is False.

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
sqrt(252), risk-free 0), trades, win rate, average win/loss, profit factor, total
commission, buy-and-hold return (first open to last close).

## Not implemented (ideas)

Shorting, ADX regime filter, volume-based partial fills, random latency,
multi-instrument portfolio, walk-forward parameter selection. 
