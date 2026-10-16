# Strategies from the literature implemented here

Short notes on the published sources behind the rule-based strategies in
`algotrader/intraday.py` and `algotrader/meanrev.py`. These papers are not openly
licensed, so only summaries and links are kept. Numbers quoted are the authors'
claims, not reproduced by this code; our own results are in `results/summary.md`.

## Market intraday momentum (Gao, Han, Li & Zhou, 2018)

- *Journal of Financial Economics*, 2018:
  <https://profiles.wustl.edu/en/publications/market-intraday-momentum/>
- SPY, 1993 to 2013: the first half-hour return (previous close to 10:00) predicts the
  last half-hour return. Stronger on volatile, high-volume and macro-news days; also
  seen in ten other ETFs. Explained by infrequent rebalancing and late-informed trading.
- Trading rule: long (short) the last half hour when the first half hour was up (down).
  A 2014 draft reported about 6% a year gross, before costs.
- **Here:** `IntradayMomentumStrategy`, long side only, optional threshold on the first
  half-hour return and optional confirmation from the 15:00 to 15:30 return.

## Beat the Market: intraday momentum for SPY (Zarattini, Aziz & Barbon, 2024)

- Swiss Finance Institute Research Paper 24-97:
  <https://www.sfi.ch/de/publications/n-24-97-beat-the-market-an-effective-intraday-momentum-strategy-for-s-p500-etf-spy>
- "Noise area": for each minute of the day, the average absolute move from the open
  over the last 14 days. Breaking out of the band (around the open or previous close)
  signals an intraday trend. Positions checked every 30 minutes, trailing stop at the
  band or the session VWAP, flat at the close. Reported 2007 to early 2024: about 19.6%
  a year net, Sharpe 1.33, with $0.0035/share commission and $0.001 slippage and
  volatility-targeted leverage. One informal re-implementation found weaker results.
- **Here:** `NoiseAreaStrategy`, long side only, fully invested instead of volatility
  targeting, IBKR commissions.

## Opening-range breakout (Zarattini & Aziz, 2023)

- "Can Day Trading Really Be Profitable?", SSRN 4416622.
- Trade in the direction of the first 5-minute candle from the second candle's open,
  stop at the other side of the first candle, target 10R or the close. Reported to work
  on "stocks in play" (unusually high opening volume). The paper charges commission
  but no spread or slippage and has no out-of-sample period; one independent
  replication on index CFDs reproduced the gross edge but found nothing net of costs
  (<https://www.mql5.com/en/blogs/post/776235>).
- **Here:** `OpeningRangeBreakoutStrategy`, long side only, opening range of 5, 15 or
  30 minutes, optional relative-volume filter, no profit target.

## Overnight versus intraday returns

- Cliff, Cooper & Gulen (2008), "Return differences between trading and non-trading
  hours: like night and day" (SSRN working paper); Lou, Polk & Skouras (2019), "A tug of war:
  overnight versus intraday expected returns", *Journal of Financial Economics* 134(1).
- Over long US samples most of the equity premium was earned overnight (close to open)
  rather than during the session.
- **Here:** `OvernightStrategy`: buy one minute before the close, sell at the next open,
  optional trend filter. One round trip per day, so costs are the question.

## RSI(2) mean reversion (Connors & Alvarez, 2008)

- *Short Term Trading Strategies That Work* (book). Buy when the 2-day RSI is very low
  (for example below 10) while above the 200-day average; sell when the close rises
  above the 5-day average. Practitioner rule, widely backtested on index ETFs.
- **Here:** `RsiReversionStrategy` on daily bars, threshold, exit average and trend
  filter tuned on the training period.

## Time-series momentum (Moskowitz, Ooi & Pedersen, 2012)

- "Time series momentum", *Journal of Financial Economics* 104(2), 2012.
- Across 58 futures and forwards (equity indices, bonds, currencies, commodities), an
  asset's own past 12-month return predicts its next month's return; the strategy is
  long assets with positive and short those with negative trailing returns, each scaled
  to a constant volatility (40% a year in the paper). Strongest at 1 to 12 month
  lookbacks; it partly reverses after a year.
- **Here:** `TimeSeriesMomentumStrategy` (`momentum.py`), long side only, lookback 63,
  126 or 252 days, optionally skipping the last 21 days, with or without volatility
  targeting (`vol_target` overlay).

## Turn of the month (Lakonishok & Smidt, 1988; McConnell & Xu, 2008)

- Lakonishok & Smidt, "Are seasonal anomalies real? A ninety-year perspective",
  *Review of Financial Studies* 1(4), 1988; McConnell & Xu, "Equity returns at the turn
  of the month", *Financial Analysts Journal* 64(2), 2008.
- US stock returns are concentrated in a short window from the last trading day of a
  month to the first three of the next; McConnell & Xu found the effect in most of 35
  countries and not explained by month-end cash flows alone.
- **Here:** `TurnOfMonthStrategy` (`momentum.py`): hold the last 0 to 3 trading days and
  the first 1, 3 or 5. The last month of the data is treated as incomplete (no
  month-end signal), since otherwise the data's final day would look like a month end.

## Volatility targeting (Moreira & Muir, 2017)

- "Volatility-managed portfolios", *Journal of Finance* 72(4), 2017.
- Scaling exposure by the inverse of recent realised variance raised Sharpe ratios for
  the market and several factors, because volatility is persistent but does not predict
  returns proportionally. Later work (Cederburg et al., 2020) found the gains weaker
  out of sample once the scaling is estimated in real time.
- **Here:** `VolTargeted` overlay (`overlays.py`): each entry's size is
  min(1, target / realised 20-day volatility); applied to SMA, RSI(2) and time-series
  momentum as a tuned option.

## Multiple testing (Bailey & López de Prado, 2014)

- "The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and
  non-normality", *Journal of Portfolio Management* 40(5), 2014.
- When N configurations are tried, the best one's Sharpe is inflated; under zero edge
  its expected value is about SE * ((1 - gamma) z(1 - 1/N) + gamma z(1 - 1/(N e))).
- **Here:** `results/summary.md` prints that value for the total number of
  configurations tried, as a bar a result has to clear before it means anything.

## Cross-sectional momentum and short-term reversal (Jegadeesh & Titman, 1993; Jegadeesh, 1990)

- Jegadeesh & Titman, "Returns to buying winners and selling losers", *Journal of
  Finance* 48(1), 1993: US stocks ranked on 3 to 12-month returns; buying the top decile
  and selling the bottom earned about 1% a month over the following months (1965-1989).
- Jegadeesh, "Evidence of predictable behavior of security returns", *Journal of
  Finance* 45(3), 1990 (and Lehmann, 1990): the previous month's (or week's) losers
  outperform its winners over the next month.
- **Here:** `CrossSectionalMomentum` and `ShortTermReversal` (`portfolio.py`), long-only
  top/bottom N, and a long-short momentum book (`short_n`); one portfolio over the
  universe, monthly or weekly rebalancing.

## Pairs trading (Gatev, Goetzmann & Rouwenhorst, 2006)

- "Pairs trading: performance of a relative-value arbitrage rule", *Review of
  Financial Studies* 19(3), 2006: pairs matched on minimum distance between normalised
  prices, opened when prices diverge by two standard deviations and closed on
  convergence, earned excess returns of up to about 11% a year (1962-2002), declining in
  later years.
- **Here:** `PairsStrategy` (`portfolio.py`) on six pairs fixed in advance (same
  industry or near-identical ETFs) rather than searched for in the data, z-score of the
  log price ratio, entry/exit thresholds tuned on training data.

## Machine-learning return prediction (Gu, Kelly & Xiu, 2020)

- "Empirical asset pricing via machine learning", *Review of Financial Studies* 33(5),
  2020: tree ensembles and neural networks on about 900 stock characteristics predicted
  monthly US stock returns better than linear models, with long-short decile portfolios
  reaching high Sharpe ratios (mostly from small, illiquid stocks and before costs).
- **Here:** `CrossSectionalML` (`portfolio.py`): a much smaller version with nine
  price-based features ranked across tickers, gradient boosting, weekly top-N (and
  optional bottom-N short) portfolios, refitted at each walk-forward fold.
