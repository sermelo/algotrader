# Roadmap: research-based priority list

Created 2026-10-16. **Keep this file up to date while implementing**: mark items done
(with the commit), add findings, re-rank when new evidence appears. The ranking follows
the published evidence, not preference. Numbers are the authors' claims (from papers,
abstracts or reviews) unless marked as ours.

## How items are ranked

1. **Evidence strength:** peer-reviewed; replicated by independent studies; survives
   trading costs; survives after publication; holds in liquid large caps (our universe
   has no microcaps).
2. **Whether we can test it meaningfully** with the data we have or can get. An item
   whose evidence is strong but needs data we lack is ranked behind the data task
   that unlocks it.

Background evidence used throughout:

- **Most published factors replicate, but are weaker than claimed.** Jensen, Kelly &
  Pedersen, "Is there a replication crisis in finance?", *Journal of Finance* 78(5),
  2023: 82% of 153 factors replicate (Bayesian), above 75% in 10 of 13 themes; the
  exceptions are seasonality, leverage and size; large/mega caps 81.5%/77.3%. Hou, Xue
  & Zhang, "Replicating anomalies", *RFS* 2020: with value-weighted returns 65% of 452
  anomalies fail t > 1.96; replication rates by category: investment 73.7%, momentum
  63.2%, profitability 44.3%, value 42%, intangibles 25.2%, trading frictions 3.8%.
- **Returns fall after publication.** McLean & Pontiff, *Journal of Finance* 2016:
  about 58% lower post-publication, more so in large, liquid stocks.
- **Costs kill high turnover.** Novy-Marx & Velikov, "A taxonomy of anomalies and their
  trading costs", *RFS* 2016: most anomalies with one-sided monthly turnover below 50%
  stay significant after costs; almost none above it. Best simple fix: a buy/hold
  spread (keep holding names you would not buy now). Our results agree: every intraday
  strategy loses to a 2 bp half-spread.
- **Machine-learning profits concentrate where we cannot trade.** Avramov, Cheng &
  Metzker, *Management Science* 2023: deep-learning signals earn mostly in microcaps,
  distressed stocks and high-volatility periods, and deteriorate after costs.
- **Our test has little power.** 3.3 years of walk-forward: the standard error of an
  annual Sharpe is about 1/sqrt(3.3) = 0.55, so a true Sharpe of 0.5 above the
  benchmark is indistinguishable from zero, and 288 configurations tried put the luck
  bar at 1.59 (Bailey & Lopez de Prado 2014). **This is the main reason data comes
  first.**

## Constraints for trading it live (from the investment-portfolio session, 2026-10-16)

sergio would trade through Interactive Brokers Ireland (IBKR Pro; Spanish tax resident).
Unverified figures, see https://www.interactivebrokers.com/en/pricing/commissions-stocks.php.

- **Costs:** the backtester models IBKR Pro Fixed ($0.005/share, $1 minimum, fees
  included; no 1% cap, irrelevant above penny stocks) + 2 bp half-spread (stress 5 bp).
  IBKR Pro Tiered ($0.0035/share, $0.35 minimum, plus exchange ~$0.003/share when taking
  liquidity, clearing ~$0.0002/share, SEC and FINRA fees on sales) is about 30% cheaper
  per order at our sizes (e.g. 66 shares of a $150 stock: Fixed $1.00, Tiered about
  $0.56 to buy and $0.77 to sell), still small next to the spread. Base round trip ~6 bp
  all-in vs the session's realistic 5-10 bp: the model is in range, slightly conservative.
- **PRIIPs:** EU retail investors cannot buy US-domiciled ETFs (SPY, QQQ, IWM, DIA, TLT,
  GLD, sector ETFs). Strategies that trade them (dual momentum, multi-horizon trend,
  sector momentum, pairs on SPY/DIA and QQQ/XLK, buy-and-hold of the index ETFs) are
  not tradable as tested; UCITS ETFs (EU-listed) or individual stocks would be needed.
  Using SPY only as a signal (regime filter) or benchmark is fine.
- **Pattern day trading:** the $25k rule was removed by the SEC on 2026-06-04 but brokers
  have until 2027-10-20; if IBKR still applies it, a margin account under $25k can make
  3 day trades in 5 days (matters for the intraday strategies only).
- **Spanish two-month rule:** a loss is deferred if the same stock is bought back within
  two months: frequent trading of the same tickers changes after-tax results (results
  here are pre-tax; ask a tax adviser).
- **FX:** keep a USD balance (EUR->USD ~0.2 bp, $2 minimum per conversion); market data
  a few $ to ~$15 a month.
- **Paper trading** fills at best bid/ask with no queue or depth, so it overstates live
  results.
- **Integration (TWS API via IB Gateway + ib_async, IBC, paper first, market data, API
  limits, safety controls):** see [`docs/ibkr-integration.md`](docs/ibkr-integration.md).

## Pending tasks (keep this list current)

Updated 2026-10-17. In order; details in the sections below.

1. ~~**Finish the batched rerun**~~ **done 2026-10-17** (35 + 35 experiments, all
   fingerprints current; numbers moved by at most 0.01 except broad `xs_reversal_daily`
   0.50 -> 0.43). Committed with the code (costs, engine, intraday, two new experiments).
2. **D1 second batch**: on or after 2026-11-01 rerun `scripts/history_queue.sh` on the VPS
   for the remaining ~216 Tiingo symbols (the 500-symbol monthly quota stops it), then
   `python -m algotrader.history check`.
3. **Long-history results**: `python -m algotrader.longrun` on the VPS (19 experiments,
   deep_arb_long takes about 12 hours), `python -m algotrader.french report --results
   results/long`, `python -m algotrader.trials`, commit `results/long/` with docs.
4. ~~**S8 follow-up**: market-on-close variant on the 1-minute data~~ **done 2026-10-17**
   (`scripts/moc_residual_check.py`): MOC does not rescue it (see S8).
5. **S9 price-chart CNN** (Jiang, Kelly & Xiu): next deep-learning item, needs the long
   history; train on the VPS.
6. **S13 leftovers** (next shared-code rerun): "stocks in play" ranking for ORB, overnight
   earnings-gap filter (earnings dates from D3).
7. **Low priority**: S10 deep momentum networks on ETFs (note PRIIPs: US ETFs are not
   buyable by sergio), S11 neural ranking, D5 SIP minute data, Russell 1000 membership.
8. **Before any live trading**: paper trading per `docs/ibkr-integration.md`, stock-only
   strategies (PRIIPs), the Tiered cost model, PDT and Spanish two-month rule checks.

## Priority list

### P1. Data (unblocks everything else)

| # | Task | Why (evidence / need) | Source options | Status |
|---|---|---|---|---|
| D1 | **Long daily price history** (20+ years) for the broad universe, ideally including delisted names | Power (above); seasonality, 12-month momentum, low-vol and trend need many years; the papers tested decades | Tiingo EOD (free plan, serves delisted stocks) + Quandl WIKI (Kaggle, to 2018); `history.py`. Free route decided by sergio 2026-10-16 | **in progress**: WIKI 673 of 818 members; Tiingo first batch ~500 symbols (437 on 2026-10-17 morning), remaining ~216 after the quota reset on 2026-11-01 |
| D2 | ~~Fama-French + momentum daily factors~~ **done** (Kenneth French data library, free, from 1926) | Regress our returns on market, size, value, profitability, investment, momentum to see whether "alpha" is just factor exposure; long-history check of factor strategies; inputs for factor momentum (S5) | mba.tuck.dartmouth.edu Data Library (CSV) | **done**: `french.py`, see M1 |
| D3 | ~~Point-in-time fundamentals and filing dates~~ **done** (`edgar.py`) | Unlocks value, profitability, investment (S3), the best-replicating categories after momentum; filing dates give earnings dates for PEAD (S7) and the overnight earnings filter | `data.sec.gov` companyfacts API / nightly bulk zip, free, User-Agent required, about 10 requests/s. Must key each fact to its filing date (not the period end) and keep amendments in order | **done**: 748 of 818 members since 2008 mapped to a CIK (590 by ticker, 158 by name); members with XBRL facts 82% in 2008, 90% in 2010, 96-98% from 2020; 1.54M facts for 722 companies; 430k earnings releases (8-K item 2.02) |
| D4 | **Bigger, point-in-time universe** (e.g. historical S&P 500 / Russell 1000 members) | Cross-sectional strategies need breadth (papers use hundreds to thousands of stocks; deep stat arb used about 550 large caps) | Index membership lists (free ones are incomplete; paid ones exist); combine with D1 | **partly done**: point-in-time S&P 500 from GitHub `hanshof` (daily lists 1996 to 2025-08-23; after that the last list is carried forward), used by `longrun.py`. Russell 1000 not done |
| D5 | Consolidated (SIP) minute data | Only if an intraday item gets promoted again: IEX volume is a small share of the tape; also 10 years of intraday history instead of 3.3 | **Alpaca free Basic plan** (checked 2026-10-17): historical SIP bars and trades since 2016, only the latest 15 minutes restricted. Unverified: delisted symbols, rate limits, terms of use; needs a free Alpaca account (sergio) | later (free source found) |

### P2. Strategies

| # | Strategy | Evidence | Needs | Status |
|---|---|---|---|---|
| S1 | **Cross-sectional momentum (12-1 month), long-short and long-only**, with a Novy-Marx-Velikov buy/hold spread to cut turnover | Strongest replication (HXZ 63%, JKP); low turnover, survives costs (NMV); decays after publication (McLean & Pontiff). Lou, Polk & Skouras (*JFE* 2019): momentum profits are earned overnight. **Ours:** the only strategy consistent on both universes (alpha t 1.49 / 1.52) | Buffer done (`xs_momentum_buffer*`, `buffered.py`): core 1.18 / 1.05, broad 0.91 / 0.76, no factor alpha; little cost to save at monthly turnover. Retest on D1 + D4 | buffer done; retest on long history |
| S2 | **Trend following / time-series momentum across asset classes** (1, 3, 12-month signals combined, 10% vol target) | Hurst, Ooi & Pedersen, "A century of evidence on trend-following investing", *JPM* 2017 (67 markets, 1880-2016); Moskowitz, Ooi & Pedersen, *JFE* 2012. Evidence is on diversified futures; on single stocks weaker. **Ours:** `tsmom_daily` 1.32 core / 1.04 broad, alpha about 0 | Multi-horizon done (`trend_multi*`, `trend.py`): long-short 0.66 / 0.37, long or cash 0.83 / 0.86, no alpha. Retest on long history (Tiingo ETFs from the 1990s/2000s) | done; retest on long history |
| S3 | **Value, profitability/quality, investment** (book-to-market, gross profitability, asset growth) | HXZ: investment 73.7%, value 42%, profitability 44% value-weighted; JKP: most themes replicate, also in large caps; low turnover, survive costs (NMV) | D3, D1 | **code done** (`fundamentals.py`, `longrun.py` value/profitability/asset_growth); results wait for D1 |
| S4 | **Return seasonality** (same calendar month in past years; Heston & Sadka, *JFE* 2008; Keloharju, Linnainmaa & Nyberg, *JF* 2016: about 13% a year) | Strong original papers, holds in developed markets (Li, Zhang & Zheng); but seasonality is one of the three themes that replicate poorly in JKP | D1 | **code done** (`seasonal.py`, `seasonality*_long`); results wait for D1 |
| S5 | **Factor momentum** (Ehsani & Linnainmaa, *JF* 2022: factors after a positive year earn 51 bp/month vs 6 bp after a negative year) | Peer-reviewed; later replications dispute that it subsumes stock momentum globally | D2 for evidence; tradeable only through our own factor portfolios (S1, S3, low vol). **Checked 2026-10-17 on Ken French's 6 factors (Mkt-RF, SMB, HML, RMW, CMA, Mom), monthly:** time-series factor momentum (long if past 12-month return > 0, else short) vs holding the six equally: alpha +0.1%/yr (t 0.10) 1964-2018, +2.2%/yr (t 1.03) 2019-2026. No value added with these factors (the paper used ~20) | deprioritised |
| S6 | **Low volatility / defensive** | Ang et al. 2006; Frazzini & Pedersen, *JFE* 2014; but Novy-Marx & Velikov, "Betting against betting against beta", *JFE* 2022: BAB returns come from microcaps, costs cut about 60%, no net alpha vs FF5. Defensive tilts are large-cap (Novy-Marx 2016). **Ours:** `low_vol_daily` 1.24 core | Implemented; retest on D1, against D2 factors (likely just profitability/investment exposure) | done (retest later) |
| S7 | **Post-earnings announcement drift** | Classic and peer-reviewed, but Martineau (2022): gone in non-microcaps since about 2006; some recent papers revive it with multi-quarter surprises (strongest in large caps), a point-in-time S&P 500 study loses after costs | D3 | **code done** (`pead.py`, `pead*_long`): surprise = announcement return (Brandt et al. 2008), since XBRL EPS is filed after the release; results wait for D1 |
| S8 | **Deep learning statistical arbitrage** (Guijarro-Ordonez, Pelger & Zanotti, *Management Science* 2025) | Best deep-learning evidence: daily US large caps, profitable after costs; independent replication (arXiv 2412.11432) finds even higher Sharpe, suspects overfitting/missing costs | D1 + D4 (about 500 large caps, many years), PyTorch on the VPS. **Paper's own after-cost results** (Table IX, A.X; 5 bp per trade, 1 bp per short, trained with the cost penalty, 2002-2016): PCA residuals Sharpe 0.8-0.9 (K = 1-5), negative for K >= 10; IPCA 1.0-1.2 (gross 3.4 / 4). We can do the PCA version (IPCA needs 46 characteristics). **Dry run 2026-10-17 (WIKI only, 2013 to Mar 2018, not committed): Sharpe -5.50, gross -0.31, turnover 1,455%/yr one-sided, 0/6 folds positive.** Likely cause: the paper earns close-to-close returns (trading at the close); we decide at the close and trade at the next open, so the overnight part of the residual reversal is missed; turnover then makes it worse. **Confirmed 2026-10-17** (`scripts/residual_timing_check.py`, WIKI 2010 to Mar 2018): a plain contrarian rule on the same residuals has gross Sharpe 1.57 / 0.78 / 0.76 close-to-close (lookback 1 / 5 / 30 days) but 0.46 / -0.02 / 0.36 when traded at the next open: most of the daily residual reversal happens overnight. With daily bars the only honest execution is the next open, so on the long history the strategy cannot earn what the paper reports. Trading at the close needs the signal a few minutes before it (market-on-close orders), possible only with our 1-minute data (2022 on). **MOC check 2026-10-17** (`scripts/moc_residual_check.py`, 87 broad-universe stocks, 1-minute IEX bars, residual tests Mar 2023 to Oct 2026, same contrarian rule): gross Sharpe for L = 1 / 5 / 30 days: close-to-close (paper, not tradable) 0.62 / 0.40 / -0.15; signal from 15:45 prices with a market-on-close fill 0.46 / 0.36 / -0.17 (net at 2 bp per unit turnover -0.22 / 0.04 / -0.32; turnover 363x / 160x / 62x a year); next open 1.09 / 0.92 / 0.08 (net 0.48 / 0.63 / -0.06, at 5 bp -0.39 / 0.24 / -0.24). The 15:45 snapshot keeps most of the close-to-close signal, so MOC is workable, but here there is little close-to-close reversal to capture; next-open timing is better in this period (the opposite of WIKI 2010-18), an ex-post look at 3 settings on 87 stocks, within noise (SE about 0.53). Status: deep_arb_long stays in the long runner as the honest test; no MOC experiment | blocked by execution timing; MOC checked, no edge |
| S9 | **Price-chart CNN** (Jiang, Kelly & Xiu, *JF* 2023) | Top journal, 2001-2019 out of sample, authors report transfer across markets; only an unreviewed replication; headline Sharpe ratios are equal-weighted | D1 + D4 (trained on all US stocks for decades) | todo |
| S10 | **Deep momentum networks / Momentum Transformer** (Lim, Zohren & Roberts 2019; Wood et al. 2021) | Futures only; edge survives costs only to 2-3 bp (10 bp with turnover penalty); Pollok & Robik 2026: learned policies beat simple rules in some, not all, cases | Fits current data (ETFs); extends S2 | todo |
| S11 | Neural / transformer cross-sectional ranking (Gu, Kelly & Xiu 2020; Kelly et al., NBER 2025) | Strong on the full market, weak for large caps (Avramov et al.). **Ours:** gradient-boosting ranking already lost | D1 + D4 + D3 features | low |
| S12 | Market intraday momentum (Gao et al. 2018; Baltussen et al., *JFE* 2021, 60+ futures, explained by gamma hedging) | Peer-reviewed but on futures; **ours:** last-half-hour SPY-style rule loses to costs on stocks (-3.8 Sharpe) | Shorting in the single-ticker engine; could test on SPY/QQQ only | **not doing** (2026-10-17): low evidence and our long-only version loses even before costs (gross -0.41) |
| S13 | Small items left from the first plan: ORB 10R target and "stocks in play", overnight day-of-week/earnings filters, ~~pre-holiday effect~~ (done 2026-10-17: Sharpe 0.17 / 0.07, no alpha) | ORB papers are SSRN working papers; **ours:** ORB and overnight already fail after costs (gross edges too small) | Minute data we have; earnings filter needs D3 | ORB 10R target and overnight skip-weekday **code done** (`orb_5min_100k_target`, `overnight_1min_100k_weekday`); results: ORB target 0.10 core / -0.19 broad, overnight skipping Monday nights 0.42 / 0.47 (alpha t -0.22 / -0.13), no edge. Not done: "stocks in play" ranking, overnight earnings-gap filter |
| S14 | Pretrained time-series models (Chronos, TimesFM); LSTM next-day direction | Evidence against: lose to CatBoost zero-shot (Rahimikia et al. 2025); LSTM direction has no profit after costs since 2010 (Fischer & Krauss, *EJOR* 2018) | | skip |

### P3. Method improvements (cheap, do alongside)

- M1. **Done** (`french.py`, `results/factor_alpha.md`): alpha against Fama-French 5
  factors + momentum. Finding: long-short momentum loads 0.40 (core) / 0.77 (broad) on
  the momentum factor and its alpha falls from t 1.5 to t 0.5 / 0.4, so it is the known
  momentum premium. Dual momentum on broad keeps +14.8% (t 1.47): gold and bonds, which
  the equity factors do not span. Nothing has a factor alpha t above 1.5.
- M2. **Done** (`trials.py`, `results/trials.md`): 778 configurations over core and broad,
  luck bar 1.75 over 3.3 years; the long-history count is added when `results/long/` exists.
- M3. Turnover per strategy: **done in `longrun.py`** (one-sided, per year); not yet in the main runner (would change every fingerprint).

## Data sources and costs (researched 2026-10-16)

Free (vendor or uploader claims, not yet verified by us):

| Need | Free source | Covers | Gap |
|---|---|---|---|
| D2 factor returns | Kenneth French data library | Daily FF factors + momentum, reversal, from 1926 | none |
| D3 fundamentals, filing dates | SEC EDGAR XBRL (`data.sec.gov`) | All filers, incl. delisted, with filing dates | XBRL only from about 2009-2011; we must build point-in-time handling |
| D4 S&P 500 membership | GitHub `hanshof/sp500_constituents` | Members by date from 1996 | completeness unverified |
| D1 prices, current tickers | Stooq bulk CSV; Tiingo free (500 symbols/month, 1,000 requests/day, internal use); yfinance | Decades of daily bars | **survivors only** (delisted coverage unconfirmed) |
| D1 prices, delisted | Kaggle mirrors of Quandl WIKI (about 3,000 US stocks incl. delisted, frozen March 2018); Kaggle "Huge Stock Market Dataset" (to Nov 2017, delisted unclear); Kaggle "Arandkei" delisted archive (contents unverified) | Up to 2017-2018 | names delisted 2018-2022 missing (our HF data starts March 2022); mixing sources with different adjustments |

Free is enough for D2, D3 (from about 2010) and a rough D4. For D1 a free patchwork
still leaves survivorship bias for 2018-2022 and stitching errors.

Paid options that close the gap (prices as listed on 2026-10-16; check before buying):

| Provider | Price | What it claims | Notes |
|---|---|---|---|
| **Sharadar Prices, full history** (personal use) | **$39/month or $299/year** | 25,000+ active and delisted US stocks and funds, daily from 1998; corporate actions; S&P 500 membership table | Cheapest complete fix for D1 + D4. Personal Use License (read terms before buying). 10-year history: $19/month |
| Sharadar Bundle, full history (personal use) | $69/month or $499/year | Prices + point-in-time fundamentals (from about 1990s), events, insiders | Also replaces most of D3 and gives pre-2009 fundamentals |
| EODHD All-World EOD | $19.99/month | Says 60,191 delisted US tickers, EOD history | Reused tickers can map to the wrong company; no index membership in this plan |
| Norgate Platinum (US stocks) | $346.50 / 6 months, $630 / year (no monthly) | Delisted from 1990, historical index constituents | Desktop updater software, made for trading platforms |
| Tiingo Power | $30/month | More symbols and requests | Delisted coverage not documented |
| Sierra Chart (Package 3 / 5) | about $26 / $36 a month (third-party figures) | US stocks: 1-minute bars from 2008 (tick data for the last 365 days), daily from 1980; NYSE, AMEX, NASDAQ, FINRA | Checked 2026-10-17, **declined (sergio: free sources only)**. Unconfirmed: delisted stocks, consolidated or not, export/licence terms; Windows charting app, export per chart or `.scid` files; its own daily stock data has no dividend adjustment or symbol-change handling. Alpaca's free plan covers the intraday need from 2016 |
| CRSP (WRDS) | n/a | The academic standard | Only through a university affiliation |

**One month and keep the data is not allowed by Sharadar** (terms, section 10, read
2026-10-16): on termination you must stop using the data and delete all copies,
including bulk downloads, within 30 days (an affidavit may be requested). You may keep
backtest results, models and other derived work that cannot reproduce the data. No
refunds on cancellation. So one month would only let us run every experiment inside
that month; reruns after a code change would need a new subscription, which breaks our
"results committed with the code" workflow. Other vendors' retention terms (EODHD,
Norgate) not checked yet.

**Decision (sergio, 2026-10-16, confirmed 2026-10-17): free data sources only.** Paid data only if the free
results show the survivorship gap matters; then prefer an ongoing subscription
(Sharadar Prices full history, $299/year) over a one-month download.

### D1 findings (2026-10-16, while building it)

- Accounts: Kaggle token in `~/.kaggle/access_token`, Tiingo token in
  `~/.config/tiingo/token` (both chmod 600, never print or commit). Same paths on the VPS.
- **Tiingo free serves delisted stocks** with split/dividend-adjusted daily history
  (checked: TWTR 2013-2022). Its supported-ticker list
  (`apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip`, no key needed) has
  about 7,600 US stocks that ended before 2026. Free limits (third-party, unverified):
  500 unique symbols a month, 50 requests an hour, 1,000 a day. So about 700 tickers
  take two months and must be throttled (one request per about 75 s).
- **Kaggle `marketneutral/quandl-wiki-prices-us-equites`**: 3,199 US tickers incl.
  delisted, adjusted OHLCV, to 2018-03-27 (1.8 GB CSV, 463 MB zip).
- **Universe:** point-in-time S&P 500 membership (GitHub `hanshof/sp500_constituents`,
  MIT, daily lists 1996-01-02 to 2025-08-23). Members found in Tiingo or WIKI: 74% in
  1996, 86% in 2005, 95% in 2008, 98% in 2010, 100% from 2013. Start the long
  history in 2008 (818 members since then: 42 complete in WIKI, 710 in Tiingo, 66 not
  found by ticker, mostly renames such as ABC->COR, ANTM->ELV, BLL->BALL, or failures
  before 2009 such as LEHMQ, WAMUQ, BSC).
- **Ticker identity is the main risk:** tickers are reused and renamed (Tiingo `BK` is
  now a Canadian fund; `EQR` resolves to a renamed company). Each series must be checked:
  its dates must cover the membership period, and its returns must match WIKI (before
  2018) or the HF data (from 2022) where they overlap. Unmatched tickers are logged,
  not guessed.

### Long-history runner (2026-10-16)

`longrun.py` runs 19 experiments (momentum, buffered momentum, reversal, low vol, 52-week
high, value, gross and operating profitability, asset growth, seasonality, earnings
drift, deep statistical arbitrage; long-only and
long-short) on the point-in-time S&P 500 from 2008, yearly folds from 2012. Smoke test on
WIKI only (2012 to Mar 2018, no SPY): low vol 1.20, momentum 1.02, not committed.
**Validation (2026-10-17, WIKI only, 2012 to Mar 2018, gross returns vs Kenneth French
factors, monthly correlation):** long-short momentum 0.83 with Mom (+2.8%/yr vs +3.3%),
value 0.66 with HML, asset growth 0.50 with CMA, profitability 0.33 with RMW (RMW is
operating profitability on all stocks; ours is gross profitability on S&P 500 members,
banks missing). Momentum confirms the engine and data. Profitability checked: with
operating income / book equity (RMW's definition) coverage rises from 280 to 422 stocks
and the correlation to 0.49; the rest is universe and weighting (French: all stocks,
value-weighted size-sorted portfolios). Added as `op_profitability*_long`.
**Full results need all Tiingo prices:** about 716 symbols, 500 a month on the free
plan. **Decision (sergio, 2026-10-16): wait for the free quota to reset in November**
(no paid plan). First batch of ~500 runs on the VPS on 2026-10-16/17; on or after
2026-11-01 rerun `scripts/history_queue.sh` on the VPS (it resumes, skipping saved
symbols) for the remaining ~216. Then:
`python -m algotrader.history check`, `python -m algotrader.longrun` on the VPS,
`python -m algotrader.french report --results results/long`, commit `results/long/`.

### Changes to batch into one full rerun

These touch shared code (`costs.py`, `engine.py`, `intraday.py`), so each would make every
or many existing results stale and need the ~10-hour VPS rerun. Do them together:
- ~~IBKR Pro Tiered `CostModel` (`ibkr_tiered()`) and the 1% commission cap~~ code done;
- ~~ORB 10R profit target~~ and ~~overnight day-of-week filter~~ code done (S13);
- shorting in the single-ticker engine (S12): not doing;
- still open for a later rerun: "stocks in play" ranking, overnight earnings-gap filter.

Status 2026-10-17: the done items are committed with the full rerun (`scripts/full_rerun.sh`).

## Recommended order

1. ~~D2 and M1~~ done: current "alphas" are factor exposure (see M1).
2. D1 (free route, in progress), then run the long history: the biggest gain in
   statistical power.
3. ~~S1 buy/hold spread and S2 multi-horizon trend~~ done (no edge on 3.3 years; retest on D1).
4. ~~D3~~ done; next S3 (value/profitability/investment) and S7 (PEAD), once D1 prices are
   in (Tiingo download running on the VPS; second batch after the monthly quota resets).
   Known mapping limits: 70 members unmapped (mostly acquired/renamed before 2012:
   ANTM->Elevance, ABC->Cencora, BLL->Ball, LEHMQ, WAMUQ...); CVC maps to its
   subsidiary CSC Holdings; XBRL starts 2009-2011, so fundamentals strategies start 2010.
5. ~~S4~~ code done (results wait for D1); ~~S5~~ checked, deprioritised.
6. ~~D4~~ partly done (S&P 500); ~~S8~~ built, fails with next-open execution (see S8);
   S9 (chart CNN) next among the deep-learning items.
7. S10 to S13 only if earlier results justify them (S13: pre-holiday, ORB target and
   overnight weekday done).

## Sources

- Jensen, Kelly & Pedersen 2023: https://www.nber.org/system/files/working_papers/w28432/w28432.pdf
- Hou, Xue & Zhang 2020: https://global-q.org/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf
- McLean & Pontiff 2016: https://www.hec.ca/finance/Fichier/McLean.pdf
- Novy-Marx & Velikov 2016: https://www.nber.org/system/files/working_papers/w20721/w20721.pdf
- Novy-Marx & Velikov 2022 (BABAB): https://mysimon.rochester.edu/novy-marx/research/BABAB.pdf
- Avramov, Cheng & Metzker 2023: https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4449
- Chen & Zimmermann, Open Source Asset Pricing: https://papers.ssrn.com/abstract=3604626
- Hurst, Ooi & Pedersen 2017: https://fairmodel.econ.yale.edu/ec439/hurst.pdf
- Lou, Polk & Skouras 2019: https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
- Baltussen et al. 2021: https://www3.nd.edu/~zda/intramom.pdf
- Keloharju, Linnainmaa & Nyberg 2016: https://www.nber.org/papers/w20815.pdf
- Ehsani & Linnainmaa 2022: https://quantpedia.com/price-momentum-or-factor-momentum-what-leads-what/
- PEAD debate: https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/ ;
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7489700 ;
  https://www.sciencedirect.com/science/article/abs/pii/S1057521924003922
- Deep learning sources: see `R&D/literature-strategies.md`, section "Deep learning".
- Data: https://www.sec.gov/page/sec-api-documentation ;
  https://dev.to/tradevodata/how-to-build-a-point-in-time-fundamentals-database-from-sec-edgar-and-when-not-to-2gn6 ;
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html ;
  https://www.tiingo.com/products/end-of-day-stock-price-data ;
  https://docs.alpaca.markets/docs/about-market-data-api ; https://marketparquet.com/vs/alpaca
