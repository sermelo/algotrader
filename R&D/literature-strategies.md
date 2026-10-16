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
