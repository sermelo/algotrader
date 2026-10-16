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

## Session strategies (`algotrader/intraday.py`)

Times are bar **end** times, so the rules work for any bar size that divides the
session (with 5-minute bars, "the bar ending 15:30" starts at 15:25). "Previous close"
is the last close of the previous session. Intraday strategies set `exit_at_close`
on each session's last bar, so they are flat overnight.

- **Overnight** (`OvernightStrategy`): `want_long` only on each session's
  second-to-last bar, so the engine buys at the last bar's open and sells at the
  next session's first open. On 1-minute bars that is 15:59 to 09:30. Optional trend
  filter: close above the mean of the previous `trend_days` session closes.
- **Intraday momentum** (`IntradayMomentumStrategy`, Gao et al. 2018): r1 = close at
  10:00 / previous close - 1. If r1 > `min_ret` (and, with `require_r12`, the half
  hour ending at the entry time was up), `want_long` from the bar ending at `entry`
  (15:30) to the close: buy at the next open, sell at the session close.
- **Noise area** (`NoiseAreaStrategy`, Zarattini et al. 2024): sigma(t) = mean over
  the previous `lookback` sessions of |close(t) / session open - 1| at the same
  time of day t (today excluded). Band = max(session open, previous close) *
  (1 + `band_mult` * sigma). Decisions only on bars ending at a multiple of
  `decision_minutes` from 10:00: enter if close > band; while long, stay only if
  close > max(band, session VWAP). Flat at the close.
- **Opening-range breakout** (`OpeningRangeBreakoutStrategy`, Zarattini & Aziz 2023):
  opening range = bars ending by 09:30 + `or_minutes`. If the range's last close is
  above its first open (and its volume is at least `min_rel_volume` times the mean of
  the previous `rv_lookback` sessions, when that filter is on), buy at the next open
  with `stop_price` = range low; exit at the close. Days without a complete range
  are skipped. No profit target.

Session VWAP is on the typical price (H+L+C)/3 from the session's first bar.
Defaults use `atr_mult` 50, i.e. effectively no ATR stop.

## Daily mean reversion (`RsiReversionStrategy`)

Wilder RSI(`rsi_period`) on closes. Enter when RSI < `entry` (and close >
SMA(`trend_sma`) if set); once in, stay until close > SMA(`exit_sma`). Decided on the
close, filled at the next open. No stop by default.

## Gradient boosting (`GradientBoostingStrategy`, `algotrader/ml.py`)

- **Label:** 1 if close(t + `horizon`) / close(t) - 1 > `threshold`, else 0. Rows
  whose future is outside the training frame are dropped, so labels never reach into
  the test period.
- **Features:** the nine tree features plus minute of day, weekday, return since the
  session open, opening gap, distance from session VWAP, volume over its mean of the
  last 5 sessions' bars, and 1, 5 and 20-session returns (bar counts from the bar size).
- **Model:** scikit-learn `HistGradientBoostingClassifier` (`max_depth` 3, 200
  iterations, learning rate 0.05, L2 1.0, `random_state` 0), handling missing values
  natively. One model is fitted on all tickers' training bars together (`fit_many`).
- `want_long` while the predicted probability is above `p_enter`, the
  `entry_quantile` of the probabilities the model gives its own training bars (so
  about `1 - entry_quantile` of training bars are long). A relative threshold adapts
  to the label's base rate; an absolute one left the first version almost never
  trading.

## Experiments (`algotrader/experiments.py`)

- Universe: `algotrader/universe.py` (4 index ETFs, 20 large caps, 3 original tickers).
- **Walk-forward (main result):** folds start 2023-07, 2024-01, 2024-07, 2025-01,
  2025-07 and 2026-01. At each fold start F the parameters are chosen with data before
  F only and tested from F to the next fold start (the last to the end of the data).
  The six test periods are concatenated into one out-of-sample record.
- **Single split (continuity):** chosen before 2024-01-01, tested 2024-01-01 to the end
  (the same choice as the 2024-01 fold, longer test window).
- Rule-based strategies: one full-history backtest per grid point; a window's numbers
  are that run's equity and trades inside the window (equity rebased to the value just
  before the window, trades counted by entry date). Valid because their indicators are
  causal: the run's state at F depends only on bars before F.
- Fitted strategies: fitted before F - 6 months, grid scored on F - 6 months .. F,
  chosen setting refitted on all data before F, then run on the fold.
- Selection score: Sharpe of the equal-weight portfolio of all tickers (daily equity
  normalised to 1, returns averaged daily). A grid point is eligible only if the median
  ticker made at least 5 trades in its scoring window; otherwise the best is used and
  flagged. Data after F never influences the choice at F (unit-tested).
- Costs: **base** = IBKR commission + 2 bp half-spread per side (the headline);
  **gross** = none; **stress** = IBKR + 5 bp. Fully invested (`risk` 1, `max_frac` 1),
  $10,000 per ticker unless the experiment sets `capital` (the `_100k` variants).
- Reported for the walk-forward: Sharpe with a 90% circular block-bootstrap interval
  (20-day blocks, 1,000 resamples, seed 0), return, max drawdown, alpha (annualised
  intercept) with t-statistic and beta from regressing daily returns on the
  equal-weight buy and hold of the same tickers, gross and stress Sharpe, and how many
  folds had a positive Sharpe. The summary also gives the Sharpe the best of N
  zero-edge configurations would reach by luck over the same length (Bailey & Lopez de
  Prado 2014: SE * ((1 - gamma) z(1 - 1/N) + gamma z(1 - 1/(N e))), SE = 1/sqrt(years)),
  with N the total number of configurations tried.

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
| Explicit stop | If the strategy has a `stop_price` column (not NaN on the signal bar), it replaces the ATR stop and sizing uses fill - stop. No entry if the fill is already at or below it. |
| Exit at close | On bars where the strategy's `exit_at_close` is True, an open position is closed at that bar's close (minus slippage), and no new order is placed from that bar. |
| Backtest start | `Backtester.run(bars, start=...)` computes indicators on all bars but trades and records equity only from `start`. |
| End of data | Open position closed at the last close. |

## Position size

`shares = floor(min(equity * risk / stop_distance, equity * max_frac / price))`

with `risk = 1%`, `max_frac = 10%` by default, `stop_distance = atr_mult * ATR`.
If the order plus commission exceeds cash, size is reduced until it fits.

## Costs

- Commission: $0.005 per share, $1.00 minimum per order (Interactive Brokers fixed).
- Slippage: fixed dollars per share per side, default 0 (`--slippage`).
- Spread: `spread_bps`, a half-spread in basis points of the price paid on each side
  (default 0 in the CLI; 2 bp in the experiments).

## Metrics

Total return, CAGR, max drawdown (bar-level equity), Sharpe (daily equity returns,
sqrt(252), risk-free 0), exposure (fraction of bars ending with a position open),
trades, win rate, average win/loss (break-even trades count as losses), profit
factor (gross profit / gross loss; inf if there are wins and no losses), total
commission, buy-and-hold return (first open to last close) and buy-and-hold Sharpe
(daily closes, starting from the first open).

## Not implemented (ideas)

Shorting, ADX regime filter, volume-based partial fills, random latency,
multi-instrument portfolio, walk-forward parameter selection (and walk-forward
retraining of the decision tree).
