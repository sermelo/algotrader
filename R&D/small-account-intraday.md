# Small-account intraday trading (about 1,000 euros): research notes, 2026-10-17

Question (sergio): with about 1,000 euros only one position fits, so choose the single most
promising stock (later: 2,000 euros in up to 3 positions); predict whether it rises over the next 10, 20, 30, 90, 240 (and 500) minutes;
buy and sell the same day. How to implement it? Sources are mostly broker pages, blogs and one
journal article; figures marked (check) must be confirmed with IBKR before real money.

## 1. Constraints that shape the design

- **Costs decide the horizon.** Break-even (round trip, 2 bp half-spread each side, whole shares):
  IBKR Pro Fixed ($0.005/share, $1 minimum) 0.24% on a $1,000 trade; Tiered ($0.0035/share,
  $0.35 minimum, plus pass-through fees) 0.10% to 0.19% at $500-$1,300 for a $50-$500 stock,
  0.31% at $5 (`algotrader.slot.breakeven`; tables in the report, section 7). Tiered fees for
  IBKR Ireland: $0.0005 to $0.0035 a share by monthly volume (a small account pays the top rate),
  $0.35 minimum; Fixed $0.005, $1 minimum, 1% cap; the $0-commission Lite plan is not offered to EU
  residents ([IBKR Ireland commissions](https://www.interactivebrokers.ie/en/pricing/commissions-home.php),
  [US stock costs for UK and EU residents](https://www.interactivebrokers.ie/en/index.php?f=50589)).
  A typical absolute move over h minutes is about sigma_day x sqrt(h / 390) x 0.8: for a stock with
  1.5% daily volatility 0.19% at 10 minutes, 0.27% at 20, 0.56% at 90 and 0.94% at 240. The accuracy
  needed to break even is 0.5 + cost / (2 x typical move) (symmetric moves assumed): about 63% at 10
  minutes and 53% at 240 minutes for a 0.1% cost. Horizons of 90 minutes and longer are the only ones
  where costs are a modest share of the move. The regular session is 390 minutes, so 500 minutes cannot
  be held within one day.
- **Account type and day-trade limits.** The SEC approved FINRA's Rule 4210 changes on 2026-04-14: the
  $25,000 pattern-day-trader minimum and the label are replaced by intraday margin standards (effective
  2026-06-04, brokers may switch until 2027-10-20; a margin account is reported to need $2,000)
  ([FINRA notice](https://www.finra.org/compliance-tools/weekly-archive/04152026),
  [Yahoo Finance](https://finance.yahoo.com/markets/options/articles/sec-eliminates-25-000-pattern-050417843.html);
  secondary sources disagree on details, check IBKR's current rule). With under $2,000 the likely account
  is a **cash account**: no day-trade count, but only settled cash can be used. Since May 2024 US stocks
  settle T+1, so the proceeds of a sale are usable the next day, and selling a position bought with
  unsettled funds is a free-riding violation that puts the account on 90 days of "Cash Up Front"
  ([IBKR free riding rule](https://ibkb.interactivebrokers.com/de/node/3101)). With the whole account in
  one stock this means **one round trip a day** (check with IBKR whether it settles in the morning of T+1).
- **Currency.** A euro account buying US stocks needs dollars. Manual conversion 0.002% with a $2
  minimum (0.17% of a 1,000-euro conversion), automatic conversion about 0.03% each way
  ([IBKR Ireland](https://interactivebrokers.ie/en/trading/us-stock-trading-costs-for-uk-and-eu.php),
  [The Poor Swiss](https://thepoorswiss.com/interactive-brokers-review/comment-page-4/)): convert once and keep the
  dollars (a euro-denominated 3 bp per trade would add 0.06% to a round trip). `breakeven(..., extra_bps)`
  takes it into account.
- **Tax.** Results are pre-tax; the Spanish two-month rule applies to repeated purchases of the same stock.

- **1,000 or 2,000 euros (sergio, 2026-10-17).** 2,000 euros is about $2,300 (assuming 1.15 USD per euro), above the
  $2,000 margin minimum of the new rule, so a margin account may be possible (ask IBKR whether it already applies the
  new rule; under the old one a margin account below $25,000 is limited to 3 day trades in 5 days). Break-even per
  round trip on Tiered with 2 bp half-spread a side, by position value (`breakeven`):

  | Stock price | $580 (500 euros) | $770 (667 euros, one of three slots) | $1,150 (1,000 euros) | $2,300 (2,000 euros, one slot) |
  |---|---|---|---|---|
  | $20 | 0.20% | 0.17% | 0.14% | 0.11% |
  | $50 | 0.18% | 0.15% | 0.12% | 0.09% |
  | $125 | 0.19% | 0.14% | 0.11% | 0.08% |
  | $500 | 0.18% | 0.18% | 0.11% | 0.08% |

  Three positions of 667 euros pay 0.14-0.18% each against 0.08% for all 2,000 euros in one stock: the $0.35
  minimum is per order. Three slots buy diversification (and three chances a day) at nearly twice the cost per
  percent invested; whether it pays is an empirical question for the `SlotTrader` runs (slots 1 versus 3).

## 2. Evidence that intraday prediction can pay

- Our own tests (35 experiments, 3.3 years of IEX 1-minute bars): every intraday strategy lost to costs;
  the 1-minute tree had a gross edge of 0.15-0.35 bp a trade, the 30-minute boosting model a gross Sharpe of
  0.84-1.12 and a net Sharpe of -0.39 to -0.08, with trades on every stock at 4 bp round-trip spread plus
  Fixed commissions.
- Huddleston, Liu & Stentoft, "Intraday market predictability: a machine learning approach", *Journal of
  Financial Econometrics* 2023: 5-minute S&P 500 index returns, regularised linear and tree ensembles,
  Sharpe 0.98 after costs ([abstract](https://academic.oup.com/jfec/article-abstract/21/2/485/6400345)); the
  authors' appendix finds little economic gain on adjusted measures and tiny out-of-sample R-squared at 15
  minutes ([appendix](https://public.econ.duke.edu/~boller/Papers/HFML_Sup_App.pdf)). Index, not single stocks,
  and 5 minutes, so not a direct test of this plan.
- Single-stock minute-bar studies mostly report gross results or accuracy only. No source found tests
  30-240 minute single-stock horizons net of costs. Expect a small edge, if any.
- What is new here compared with the failed tests: (a) one trade a day, chosen as the best of hundreds of
  stocks (the top of a cross-section has a larger expected move than the average), (b) horizons of 90-240 minutes,
  (c) the cheaper Tiered plan, (d) a trade only when the predicted edge clears its own cost hurdle, (e) many
  more years of data.

## 3. Choosing one trade: confidence

- **Meta-labeling** (Lopez de Prado, *Advances in Financial Machine Learning*, 2018): a primary model
  proposes a trade, a secondary model estimates the probability the trade works and decides trade or skip and
  the size ([overview](https://en.wikipedia.org/wiki/Meta-Labeling)). A critic argues it cannot beat a well
  trained end-to-end model and can leak when both models share training data
  ([QuantConnect](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)):
  train the two on different periods.
- **Winner's curse**: picking the top of many noisy predictions inflates its observed score
  ([Andrews, Kitagawa & McCloskey, arXiv 2411.18569](https://arxiv.org/pdf/2411.18569)). With one pick out of
  hundreds, the predicted edge of the top stock overstates its true edge. Remedies used here: calibrate the
  predicted edge on an untouched later year (regress realised on predicted return, shrink by the slope), demand
  a margin over the cost hurdle (`min_margin`), and judge only on the walk-forward out-of-sample trades.
- Simplest scorer first: one regression per horizon on features of the last minutes (returns over 5, 15, 30, 60
  minutes and since the open, the opening gap, realised volatility, volume against its usual level at that time,
  distance from the session VWAP, the stock's move against SPY, time of day, the previous day's return), target =
  gross forward return over the horizon scaled by volatility. Gradient boosting (`ml.py`
  already has the pattern); a CNN on minute charts (`chartcnn.py` with d = 60 bars) later if boosting shows an edge.

## 4. Implementation plan

1. **Engine (done)**: `algotrader/slot.py`: `breakeven`, `SlotTrader` (one position, highest edge that clears
   its hurdle, hold `hold_minutes` or to the close, one trade a day, Tiered or Fixed costs, `extra_bps`).
2. **Minute features from the Hugging Face minute bars** (consolidated, 2008-Mar 2026; the adjustment for splits
   does not matter inside one day): one pass over the monthly files like `scripts/hf_daily.py`, for S&P 500
   members, writing for each stock, day and decision minute (every 10 or 30 minutes from 10:00) the features and the
   forward returns for 10, 20, 30, 90 and 240 minutes (and the session close). Small output, VPS, about 2-3 hours.
3. **Model**: walk-forward by year (fit on earlier years, a later untouched year to calibrate), one model per
   horizon, edges for all members at each decision minute.
4. **Evaluation** with `SlotTrader`: net result per trade and per year, trades a year, win rate, t-statistic of the
   mean net return per trade (about 4,600 days available: the standard error of the mean is about 0.015% if the
   per-trade spread is 1%, against a hurdle of 0.1-0.2%), against baselines (random stock, top-1 of a random
   score, buy SPY for the same minutes), every horizon counted in the luck bar (`trials.py`).
5. **Before real money**: IBKR paper account, confirm the cash-account settlement rule and the Tiered rates, the
   first month with a 1-share position to compare real and modelled costs.


## 5. Outcome of the tests (2026-10-18)

The plan in section 4 was run; everything is in `results/minute_*`, `algotrader/minutefeat|movestats|minutescan|minutemodel.py` and report
sections 7-10. **Verdict: stop** (the stopping rule was fixed before the model was run).

- Moves (18 Octobers, 50M windows a horizon): the typical move is 0.05% at 1 minute, 0.16% at 10, 0.28% at 30, 0.47% at 90 and 0.77% at 4 hours;
  against a 0.14% break-even, 1-5 minute holds are out; volatility concentrates moves (top tenth 5-11x the cost over 30-240 minutes).
- Signals: the only consistent feature effect is short-term reversal (rank correlation -0.02 to -0.035, 2-6 bp for the best pick); 6,264 simple
  picking rules: none beats luck after costs.
- Model (walk-forward gradient boosting, volatile tenth, 2013-2026): skill about +0.02 rank correlation; 0 of 12 pre-registered tests clear the cost.
- The best-of-day view (+39 bp net, t 11.7) was a hindsight mirage; the first-passage rule a trader could follow gives about 0 net (-3 to +1 bp).
- Lesson: evaluate the rule a trader could follow; choosing the trade time with hindsight about which decision of the day has the highest
  prediction selects favourable price paths, because later predictions carry information about what the price did after the entry.
- Remaining: a faint tail at high thresholds (7-28 trades a year, +38 to +115 bp gross) that only forward paper trading could confirm.
