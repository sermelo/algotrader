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
  filter: close above the mean of the previous `trend_days` session closes; optional
  `skip_weekdays` (0 = Monday .. 4 = Friday): no hold on nights starting on those days.
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
  are skipped. Optional `trend_days`: only when the previous session's close is above
  the mean of the `trend_days` closes before it. Optional `profit_r` (the paper's 10R):
  take profit at entry + profit_r x (entry - range low).

Session VWAP is on the typical price (H+L+C)/3 from the session's first bar.
Defaults use `atr_mult` 50, i.e. effectively no ATR stop.

## Time-series momentum and calendar (`algotrader/momentum.py`, daily bars)

- **Time-series momentum** (`TimeSeriesMomentumStrategy`, Moskowitz et al. 2012):
  long while close[t - `skip`] / close[t - `lookback`] - 1 > `min_ret`.
- **Turn of the month** (`TurnOfMonthStrategy`, Lakonishok & Smidt 1988): trading days
  are numbered 1, 2, ... from each month's first and -1, -2, ... from its last.
  `want_long` on the closes of days -(`days_before` + 1) .. `days_after` - 1, so the
  engine buys at the open of day -`days_before` and sells at the open of day
  `days_after` + 1. Uses which days have bars (the exchange calendar, known in advance),
  not prices; the last month of the data has no month-end positions (it may be
  incomplete).

- **Pre-holiday** (`PreHolidayStrategy`, `calendar_effects.py`; Ariel 1990): a holiday is
  a weekday without a session (business-day gap between bars). Long during the
  `days_before` sessions ending with the last one before the holiday and the
  `days_after` sessions after it: bought at the open of the first, sold at the open
  after the last. Uses only the session calendar; the last bar is never pre-holiday.

## Overlays (`algotrader/overlays.py`)

Wrappers around any strategy, selected by parameters in an experiment's grid:

- `vol_target` (+ `vol_lookback`, default 20): `VolTargeted` adds `size_frac` =
  min(1, target / realised vol), the realised vol being the annualised standard
  deviation of the last `vol_lookback` daily close-to-close returns. The engine caps
  the position at `size_frac` of equity.
- `regime_days`: `MarketRegime` keeps `want_long` only while SPY's session close is
  above its `regime_days` mean (entries blocked, open positions exited otherwise).
- Both use only sessions finished by the bar's close: on daily bars the same day's
  close, on intraday bars the previous session's.

## Portfolio strategies (`algotrader/portfolio.py`)

- `PortfolioBacktester` runs ONE long-only portfolio over all tickers on daily bars.
  A cross-sectional strategy's `weights(closes)` gives target weights per day from
  closes up to that day. Orders are placed only when the weights change, at the next
  day's open (sells first, buys scaled down if cash is short), whole shares, same
  `CostModel`. A ticker whose data ends is sold at its last close. Equity is marked
  to each day's close (last known close for tickers without a bar).
- **Cross-sectional momentum** (`CrossSectionalMomentum`, Jegadeesh & Titman 1993):
  equal weight in the `top_n` tickers by return from `lookback` to `skip` days ago,
  rebalanced at each month (or week/day) end among tickers trading that day.
- **Short-term reversal** (`ShortTermReversal`, Jegadeesh 1990): equal weight in the
  `top_n` tickers with the lowest `lookback`-day return.
- **Shorting** (`allow_short=True`): weights may be negative (whole shares, rounded
  towards zero). Short sale proceeds go to cash; equity = cash + signed market value.
  Gross exposure (sum of |weights|) is scaled down to `max_gross` (1.0 in the
  experiments, so a long-short book is 50% long and 50% short). Shorts pay
  `borrow_bps` a year (50 bp in the experiments, 0 in the gross variant) on their
  market value, charged at each close as 1/252 of the annual rate. A short in a ticker
  whose data ends is bought back at its last close. No margin calls or recalls.
- **Long-short momentum**: `CrossSectionalMomentum(short_n=N)` also shorts the N
  tickers with the lowest score (weights +1/top_n and -1/short_n before the gross cap).
- **Pairs** (`PairsStrategy`, after Gatev et al. 2006): for fixed pairs chosen in
  advance on economic grounds (XOM/CVX, JPM/BAC, KO/PEP, SPY/DIA, QQQ/XLK, AAPL/MSFT;
  pairs missing a ticker are skipped), z = (log(a/b) - mean) / std over `lookback`
  days. Enter when |z| > `entry_z` (long the cheap leg, short the rich one, 0.5 /
  n_pairs per leg), exit when |z| < `exit_z`.
- **Cross-sectional ML** (`CrossSectionalML`, in the spirit of Gu, Kelly & Xiu 2020):
  daily features (returns over 1, 5, 21, 63, 126 days, 21-day volatility, distance
  from the 50 and 200-day means, RSI(14)), each ranked across tickers per day. A
  `HistGradientBoostingRegressor` (depth 3, 200 iterations, learning rate 0.05, L2
  1.0) predicts the `horizon`-day forward return minus that day's cross-sectional mean;
  rows whose future lies after the training data are dropped. At each period end hold
  the `top_n` highest predictions (and short the `short_n` lowest). In the runner it is
  fitted before F - 6 months for tuning and refitted before each fold start F.
- **Factor portfolios** (`algotrader/factors.py`), equal weight, monthly:
  `LowVolatility` holds the `top_n` tickers with the lowest standard deviation of daily
  returns over `lookback` days (Ang et al. 2006); `High52Week` the `top_n` with the
  highest close / max close of the last `lookback` days (George & Hwang 2004);
  `SectorMomentum` the `top_n` of the nine sector ETFs (XLF XLK XLE XLV XLI XLP XLY XLU
  XLB, only in the broad universe) by return from `lookback` to `skip` days ago
  (Moskowitz & Grinblatt 1999); `DualMomentum` (Antonacci) ranks SPY, QQQ, IWM, GLD by
  `lookback`-day return, holds the best `top_n` (1/`top_n` each) only if their return
  is positive and puts the rest in `safe` (TLT, or cash if None).
- **Buffered momentum** (`BufferedMomentum`, `algotrader/buffered.py`; Novy-Marx &
  Velikov 2016 buy/hold spread): at each period end keep held names still ranked within
  `keep_n` (and trading), keep at most `top_n` of them (best first), then add the best
  others until `top_n` are held; equal weight. With `short_n`, the same on the bottom
  ranks with buffer `short_keep` (default 2 x `short_n`), weights -1/`short_n`.
- **Multi-horizon trend** (`MultiHorizonTrend`, `algotrader/trend.py`; Hurst, Ooi &
  Pedersen 2017): for each ETF of SPY QQQ IWM DIA TLT GLD and the nine sector ETFs that
  the universe has, signal = mean of sign(close / close h days ago - 1) for h = 21, 63,
  252 (none until 252 days of history); weight = signal x `target_vol` / annualised
  volatility of daily returns over `vol_lookback` days, divided by the number of assets
  with a weight; `long_only` sets negative signals to 0. Monthly; gross capped at 1.
- In the experiments a portfolio starts with $100,000 in total; the activity rule
  counts rebalance days.

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

- Universes (`algotrader/universe.py`, `--universe`): **core** (4 index ETFs, 20 large
  caps, 3 original tickers; results in `results/`) and **broad** (core plus 11 sector,
  bond and gold ETFs and 69 tickers from a seeded random sample of the provider's
  symbols, filtered on March 2022 data only: first bar before April 2022, price >= $5,
  >= 100 traded minutes a day; results in `results/broad/`). Broad includes names that
  stop trading (acquisitions, renames), which leave the portfolio at their last bar.
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

## Long history (`algotrader/longrun.py`)

- Universe: S&P 500 members on each day (daily membership lists, forward-filled between
  list dates); `PointInTime` hides the closes of non-members from the strategy, so it
  can only buy members; held stocks that leave are sold at the next rebalance; a stock
  whose data ends is sold at its last close. Prices: Tiingo if its series did not fail
  `history check`, else the adjusted Hugging Face series (`hfdata.build_source`: WIKI's
  split and dividend events where WIKI has the ticker, splits found from prices and no
  dividends after WIKI's last day; `--no-hf` skips it), else Quandl WIKI (adjusted OHLCV).
- Walk-forward: folds start every January from 2012 to the last year of data; at each
  fold start the grid point with the best portfolio Sharpe on all earlier data (and at
  least 3 rebalance days) is chosen and run until the next fold. One full-history run
  per grid point, sliced (rule-based strategies on causal signals).
- $1,000,000 portfolio, same costs as the main experiments (IBKR + 2 / 0 / 5 bp), shorts
  pay 50 bp a year, gross exposure capped at 1. Turnover = traded value / 2 / mean
  equity per year. Alpha/beta against SPY.
- **Fundamental sorts** (`fundamentals.py`): book-to-market = latest stockholders'
  equity (10-K/10-Q) / (latest shares outstanding x unadjusted close), negative equity
  excluded; gross profitability = latest annual gross profit (reported, else revenue -
  cost of revenue; 330-400-day periods) / latest annual total assets; asset growth =
  latest annual assets / previous annual assets - 1; operating profitability (RMW-style)
  = latest annual operating income / latest annual stockholders' equity (positive only). A value counts from the day after
  its filing date; for each date the value with the latest period end known so far is
  used (an amendment of that period replaces it from its own filing date; older periods
  reported later as comparatives do not).
- **Earnings drift** (`pead.py`): reaction day = the 8-K (item 2.02) acceptance day, or
  the next trading day if accepted at or after 16:00 New York time; EAR = (1+r_d)(1+r_d+1)
  - 1 minus the same for the market (SPY, or the equal-weight mean of the stocks),
  known at the close of d+1 and kept `hold_days` trading days; top/bottom N by EAR.
- **Seasonality** (`seasonal.py`): at each month end, score = mean return of the coming
  calendar month in the previous `years` years (at least `min_years`); top/bottom N.

## Price-chart CNN (`algotrader/chartcnn.py`, long history)

Jiang, Kelly & Xiu (2023). **Image**: the last `d` = 20 days as a 64 x 60 black-and-white chart,
3 pixels a day: the left pixel the open, the middle column the high-low line, the right pixel
the close; a 20-day moving average of the closes (using closes from before the window) as one
row per day; volume bars in the bottom 12 rows. Prices are divided by the window's first close
and scaled to the window's own minimum and maximum, volume to its maximum. **Label**: 1 if the
close `h` = 20 days later is above the close on the chart's last day. **Model**: three blocks of
conv (5 x 3), batch norm, leaky ReLU, 2 x 1 max pool (channels 16, 32, 64), dropout 0.5, one
logit; Adam, learning rate 1e-3, batches of 128, at most 5 epochs, early stopping (patience 2) on
the last 20% of the sampled days (a time block). **Fitting**: every January on the previous 6
years (from 2008 data for 2013), charts sampled every 20 trading days (non-overlapping), only S&P 500
members of that day, only charts whose label was known before 1 January. **Scores**: probability of
a positive return for every member at each month end of the year, from data up to that day.
**Portfolio**: `top_n` highest (and `short_n` lowest) scores, equal weight, monthly, filled at the
next open, same costs as the other long experiments; the parameters (top N, bottom N) are chosen
every January on all earlier scores. Deviations from the paper: much smaller network, higher
learning rate, fewer epochs, 6-year rolling window on S&P 500 members sampled every 20 days,
one model instead of an ensemble of five.

## Small-account intraday (`algotrader/slot.py`)

For a 1,000 to 2,000-euro account. A model gives every stock an `edge` at each decision minute: the
expected return over the next `hold_minutes`, before costs, known at that bar's close. `breakeven(cost,
price, value, extra_bps)` is the round-trip cost as a fraction of the position: commissions and fees of one
buy and one sale of whole shares plus the half-spread paid on each side, plus `extra_bps` for other costs
(e.g. currency conversion). `SlotTrader` splits the account into `slots` equal positions (each the
start-of-day equity / `slots`); at each decision minute, while a slot is free, it buys at the next bar's open
the stock with the highest edge that is at least its own `breakeven` (for the slot's value) plus `min_margin`
and that passes the price filters on the decision bar's close (`min_price`, `max_price`, `max_cost`: a ceiling on
the round-trip cost, `min_invested`: whole shares must cover at least that share of the slot);
shares = the slot's money / the fill price, reduced until the commission fits; each exit is the open of the
first bar at or after entry + `hold_minutes`, or the last bar's close if the session ends first. At most
`max_trades_per_day` trades a day (default one per slot: in a cash account the proceeds of a sale settle the
next day and re-buying with unsettled money is a free-riding violation) and a stock at most once a day.
Profits are compounded; the daily equity gives the Sharpe ratio. No look-ahead (unit-tested: later prices and
later edges do not change an earlier trade). The cost model is passed in; `ibkr_tiered()` is the plan assumed
for a small account. Splitting a small account raises the hurdle: the $0.35 minimum is per order, so a
$770 position needs about 0.14% to 0.20% against 0.08% for the whole 2,000 euros in one stock.

## Combined signal and trend filter (`algotrader/combo.py`, long history)

`CombinedSignal`: at each date, the z-score across stocks (clipped to +-3) of each ingredient:
12-1 month return (`lookback` 252, `skip` 21), minus the volatility of daily returns over
`vol_lookback` 252, close / highest close of `high_lookback` 252 days, and gross profitability (the
SEC panel of `fundamentals.py`); the score is the mean of the ingredients a stock has, and a stock
needs at least `min_components` (3). Top `top_n` held (and bottom `short_n` shorted), equal weight,
monthly. `trend_days` > 0: all weights are zeroed on days SPY's close is not above its mean of the
last `trend_days` closes (known at that close, traded at the next open). `TrendMomentum` is the
same for plain momentum. Five pre-registered experiments (`longrun_combo.py`): `combo_long`,
`combo_ls_long`, `combo_price_long` (no SEC data), `combo_trend_long`, `xs_momentum_trend_long`.
Results in `results/long/` (README, "Long history").

## Mid-cap test (`algotrader/midcap.py`)

Universe: each month-end, stocks ranked 1-1,500 by trailing dollar volume (raw close x volume, median per
month, mean of 3 months, price >= $5, at least 10 trading days, SEC fund tickers excluded), minus that day's
S&P 500 members (about 1,040 names). Prices: adjusted Hugging Face daily series (no dividends after March 2018,
none for tickers WIKI lacks). A series is cut where trading stops for more than 30 calendar days, or for more
than 5 days with the open more than 2x or under 0.5x the last close; pieces are separate names with the
parent's membership. A move of more than 5x (or under 0.2x) from one close or open to the next always cuts the
series (that return is never used); days with an adjusted close below $1 are dropped (heavy
reverse splits leave absurd adjusted prices, down to 1e-36), series with a non-positive price and exchange test
symbols are dropped
(data fix added after the first run crashed on them; that run had printed only `xs_momentum_mid` Sharpe 0.26,
the strategies and the success rule are unchanged). Same walk-forward and engine as the long history; half-spread 5 bp (base), 10 bp (stress).
Pre-registered: momentum (long, long-short), short-term reversal, low volatility, 52-week high, seasonality
(long, long-short), combined price signal, with the long-history grids. Success criterion: Fama-French alpha
t >= 2 at base costs, Sharpe above the equal-weight universe, positive at stress costs. **Result (2026-10-18): fail**; best
Fama-French alpha momentum +4.2% a year, t 1.13 (README, "Mid-cap test"). With the cleaned data momentum's Sharpe is
0.78 instead of the crashed run's 0.26: the bad series had wrecked its returns.

## Stop-loss test (`algotrader/stoploss.py`, long history)

`with_stop(cls)` adds `stop` and `trail` to a monthly strategy. Each holding has a reference price: the
close on the day it was chosen (decision day), or with `trail` the highest close since then. When a close is
at or below reference x (1 - `stop`), the holding is sold at the next open and its money stays in cash until
the next rebalance day; there the strategy's own choice applies again (a stock still in its top N is bought
back with a new reference; one kept from the previous month keeps its reference). On a stop day the other
holdings get their drifted weights (value since the rebalance / portfolio value), so they are not traded
back to equal weight. `longrun_stop.py`: momentum 12-1 top 20, low volatility 252 days top 50, 52-week high
top 100 and the combined price signal top 50 (the parameters their walk-forward chose most often), each with
stop off or 10, 15, 20, 25, 30% fixed or trailing (11 configurations); the walk-forward chooses the stop each
January on all earlier data. Each level is also reported over the whole test period, with the stops' moves
from the stop to the next rebalance's close times their weight: the losses avoided and the gains given up.

## Concentration test (`algotrader/concentration.py`, long history)

Eleven long-only strategies (momentum 12-1, low volatility, 52-week high, combined price signal, combined signal,
earnings drift 21 days, seasonality, value, gross profitability, asset growth, operating profitability) at the
parameters their walk-forward chose most often, each held at N = 3, 5, 10, 20, 50 (equal weight, monthly, $1M, base and
stress costs, from 2008, reported from 2012). `RankBand(lo, hi)`: equal weight in ranks lo..hi, from the strategy's
top-hi and top-(lo-1) portfolios. `BufferBand(n, k)`: hold n names, sell a name only when it leaves the top k, fill free
places with the best-ranked names not held (3/10, 5/20, 20/50). Multi-strategy portfolios average the parts' daily
returns: all eleven at top 5 or 20, and the five (or one) with the best Sharpe on all earlier days, chosen every January.
Pre-registered rule: concentration is better for a strategy if top 3 or top 5 minus top 50 has a Fama-French alpha
t >= 2 at base costs; better in general only if that holds for most strategies. **Result (2026-10-19): not better in
general**, 2 of 11 pass (momentum top 3, asset growth top 5); README, "Long history".

## Weekly versus monthly rebalancing (`algotrader/rebalance.py`, long history)

The concentration test's strategies except seasonality (its signal is the coming calendar month), at top 5, top 20 and
top 5 kept while in the top 20, with the ranking checked at each month's last trading day or each week's; weights change
(and orders go out at the next open) only when the chosen names change. Gross, base and stress costs. Pre-registered rule:
weekly is better for a strategy if weekly minus monthly (base costs) at top 5 or top 20 has a Fama-French alpha t >= 2;
better in general only if that holds for most strategies. **Result (2026-10-20): not better**, 0 of 10 pass (README,
"Long history").

## Merger arbitrage (`algotrader/mergerarb.py`, 2008 to Mar 2026)

A risk premium (Mitchell & Pulvino 2001), not a price signal. **Deals** from SEC filings only: the target's merger proxy or
information statement (PREM14A, DEFM14A, PREM14C, DEFM14C) or a tender-offer filing (SC TO-T, SC TO-C, SC 14D9, SC14D9C),
2008 on. From each filing's full text (first 4 MB): ticker candidates ("Nasdaq: ABCD", 'under the symbol "ABCD"') and the
cash price per share named most often ("$12.50 in cash", "$12.50 per Share, net to the seller in cash"). Decision day = the
first trading day the market could react (accepted at or after 16:00 New York: next day). A filing makes a deal if exactly
one candidate ticker, on the decision day's close (raw Hugging Face daily bars), is below the offer by more than 0% and at
most 25% (offer / close - 1), costs at least $2 and has a 20-day median dollar volume of at least $1M; blank-check
companies (SIC 6770) are skipped; a ticker gets no new deal within 365 days of its previous one. No later information
(completion, delisting) is used to select deals. **Holding:** buy at the next open; sell at the last close of the price series
(the deal closed), at the next open after the deal company's 8-K with item 1.02 (termination of a material agreement) and
without items 2.01 / 3.01 (completion, delisting), or after 540 calendar days. At most `slots` deals at once, 1/`slots` of
equity each, first come (a deal arriving while all slots are full is skipped); a ticker's price series is cut where it
probably changed hands (`midcap.segments`). Cash earns the Fama-French daily risk-free rate. $1M, IBKR + 5 bp half-spread
(base), 10 bp (stress), none (gross). Raw prices: dividends paid during a deal are not counted (understates returns).
**Pre-registered (2026-10-20, before the deal list or any backtest existed):** primary `merger_arb` (20 slots), reported
alongside 10 and 40 slots (3 configurations, no tuning). Pass: the primary's Fama-French five-factor plus momentum alpha of
its excess return over the risk-free rate has t >= 2 at base costs and its mean excess return is positive at stress costs.
Also reported: Sharpe of the excess return with a 90% interval, beta, maximum drawdown, deal count, mean and median deal
return, share of deals losing more than 10%, and the returns in 2008-09 and 2020. **Result (2026-10-21): fail**: +1.1% a year over
cash, Fama-French alpha -3.1% a year (t -1.68), beta 0.38, max drawdown -39.2% (README, "Merger arbitrage").

## Factor regressions (`algotrader/french.py`)

Separate from the runner: `python -m algotrader.french report` regresses each
experiment's walk-forward daily returns (base costs) and the buy and hold of the same
tickers on the daily Fama-French five factors and momentum (Kenneth French data
library), by OLS on the days both series have. The market factor is the total market
return (Mkt-RF + RF), because the backtests earn no interest on cash. Alpha is the
intercept times 252 with its t-statistic. Output: `results/factor_alpha.md` and `.csv`
(and in `results/broad/`), with the factor files' and the module's sha256.

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
| Size fraction | If the strategy has a `size_frac` column, the position is capped at that fraction of equity (decided on the signal bar). |
| Profit target | If the strategy has a `target_r` column (on the signal bar), exit when a bar's high reaches entry + target_r x (entry - stop), at that price or the open if it gaps above. The stop is checked first when a bar touches both. |
| Exit at close | On bars where the strategy's `exit_at_close` is True, an open position is closed at that bar's close (minus slippage), and no new order is placed from that bar. |
| Backtest start | `Backtester.run(bars, start=...)` computes indicators on all bars but trades and records equity only from `start`. |
| End of data | Open position closed at the last close. |

## Position size

`shares = floor(min(equity * risk / stop_distance, equity * max_frac / price))`

with `risk = 1%`, `max_frac = 10%` by default, `stop_distance = atr_mult * ATR`.
If the order plus commission exceeds cash, size is reduced until it fits.

## Costs

- Commission: $0.005 per share, $1.00 minimum per order, at most 1% of the trade value
  (Interactive Brokers Pro Fixed; exchange and regulatory fees included).
- `ibkr_tiered()` (not used in the experiments): $0.0035 per share, $0.35 minimum, 1% cap,
  plus pass-through fees (exchange ~$0.003 and clearing ~$0.0002 per share), SEC fee on
  sales (~$20.6 per $1M) and FINRA TAF on sales ($0.000166 per share, capped at $8.30).
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

Shorting in the single-ticker engine (only the portfolio engine can short), ADX
regime filter, volume-based partial fills, random latency, "stocks in play" ranking
across tickers for the ORB, an overnight earnings-gap filter.
