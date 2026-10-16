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

## Priority list

### P1. Data (unblocks everything else)

| # | Task | Why (evidence / need) | Source options | Status |
|---|---|---|---|---|
| D1 | **Long daily price history** (20+ years) for the broad universe, ideally including delisted names | Power (above); seasonality, 12-month momentum, low-vol and trend need many years; the papers tested decades | Tiingo EOD (free plan: internal use, 1,000 requests/day; vendor claims history to 1962, a reviewer says 20-30 years); Stooq (free, fragile downloads); Alpaca (free plan; history from 2016 per a third-party page, unverified). Survivorship-free needs paid data (CRSP via WRDS, Norgate): **decision for sergio** | todo |
| D2 | **Fama-French + momentum daily factors** (Kenneth French data library, free, from 1926) | Regress our returns on market, size, value, profitability, investment, momentum to see whether "alpha" is just factor exposure; long-history check of factor strategies; inputs for factor momentum (S5) | mba.tuck.dartmouth.edu Data Library (CSV) | todo |
| D3 | **Point-in-time fundamentals and filing dates** from SEC EDGAR XBRL | Unlocks value, profitability, investment (S3), the best-replicating categories after momentum; filing dates give earnings dates for PEAD (S7) and the overnight earnings filter | `data.sec.gov` companyfacts API / nightly bulk zip, free, User-Agent required, about 10 requests/s. Must key each fact to its filing date (not the period end) and keep amendments in order | todo |
| D4 | **Bigger, point-in-time universe** (e.g. historical S&P 500 / Russell 1000 members) | Cross-sectional strategies need breadth (papers use hundreds to thousands of stocks; deep stat arb used about 550 large caps) | Index membership lists (free ones are incomplete; paid ones exist); combine with D1 | todo |
| D5 | Consolidated (SIP) minute data | Only if an intraday item gets promoted again: IEX volume is a small share of the tape | Alpaca paid plan or others; low priority | later |

### P2. Strategies

| # | Strategy | Evidence | Needs | Status |
|---|---|---|---|---|
| S1 | **Cross-sectional momentum (12-1 month), long-short and long-only**, with a Novy-Marx-Velikov buy/hold spread to cut turnover | Strongest replication (HXZ 63%, JKP); low turnover, survives costs (NMV); decays after publication (McLean & Pontiff). Lou, Polk & Skouras (*JFE* 2019): momentum profits are earned overnight. **Ours:** the only strategy consistent on both universes (alpha t 1.49 / 1.52) | Implemented (`xs_momentum*`); add buy/hold spread; retest on D1 + D4 and against D2 factors | partly done |
| S2 | **Trend following / time-series momentum across asset classes** (1, 3, 12-month signals combined, 10% vol target) | Hurst, Ooi & Pedersen, "A century of evidence on trend-following investing", *JPM* 2017 (67 markets, 1880-2016); Moskowitz, Ooi & Pedersen, *JFE* 2012. Evidence is on diversified futures; on single stocks weaker. **Ours:** `tsmom_daily` 1.32 core / 1.04 broad, alpha about 0 | Implemented on stocks; add multi-horizon version on the ETFs we have (SPY, QQQ, IWM, TLT, GLD, sectors); D1 for history | partly done |
| S3 | **Value, profitability/quality, investment** (book-to-market, gross profitability, asset growth) | HXZ: investment 73.7%, value 42%, profitability 44% value-weighted; JKP: most themes replicate, also in large caps; low turnover, survive costs (NMV) | D3 (fundamentals), D1 | todo |
| S4 | **Return seasonality** (same calendar month in past years; Heston & Sadka, *JFE* 2008; Keloharju, Linnainmaa & Nyberg, *JF* 2016: about 13% a year) | Strong original papers, holds in developed markets (Li, Zhang & Zheng); but seasonality is one of the three themes that replicate poorly in JKP | D1 (needs 10-20 years of history), prices only | todo |
| S5 | **Factor momentum** (Ehsani & Linnainmaa, *JF* 2022: factors after a positive year earn 51 bp/month vs 6 bp after a negative year) | Peer-reviewed; later replications dispute that it subsumes stock momentum globally | D2 for evidence; tradeable only through our own factor portfolios (S1, S3, low vol) | todo |
| S6 | **Low volatility / defensive** | Ang et al. 2006; Frazzini & Pedersen, *JFE* 2014; but Novy-Marx & Velikov, "Betting against betting against beta", *JFE* 2022: BAB returns come from microcaps, costs cut about 60%, no net alpha vs FF5. Defensive tilts are large-cap (Novy-Marx 2016). **Ours:** `low_vol_daily` 1.24 core | Implemented; retest on D1, against D2 factors (likely just profitability/investment exposure) | done (retest later) |
| S7 | **Post-earnings announcement drift** | Classic and peer-reviewed, but Martineau (2022): gone in non-microcaps since about 2006; some recent papers revive it with multi-quarter surprises (strongest in large caps), a point-in-time S&P 500 study loses after costs | D3 (earnings dates and EPS surprises from filings; analyst forecasts not available) | todo |
| S8 | **Deep learning statistical arbitrage** (Guijarro-Ordonez, Pelger & Zanotti, *Management Science* 2025) | Best deep-learning evidence: daily US large caps, profitable after costs; independent replication (arXiv 2412.11432) finds even higher Sharpe, suspects overfitting/missing costs | D1 + D4 (about 500 large caps, many years), PyTorch on the VPS | todo |
| S9 | **Price-chart CNN** (Jiang, Kelly & Xiu, *JF* 2023) | Top journal, 2001-2019 out of sample, authors report transfer across markets; only an unreviewed replication; headline Sharpe ratios are equal-weighted | D1 + D4 (trained on all US stocks for decades) | todo |
| S10 | **Deep momentum networks / Momentum Transformer** (Lim, Zohren & Roberts 2019; Wood et al. 2021) | Futures only; edge survives costs only to 2-3 bp (10 bp with turnover penalty); Pollok & Robik 2026: learned policies beat simple rules in some, not all, cases | Fits current data (ETFs); extends S2 | todo |
| S11 | Neural / transformer cross-sectional ranking (Gu, Kelly & Xiu 2020; Kelly et al., NBER 2025) | Strong on the full market, weak for large caps (Avramov et al.). **Ours:** gradient-boosting ranking already lost | D1 + D4 + D3 features | low |
| S12 | Market intraday momentum (Gao et al. 2018; Baltussen et al., *JFE* 2021, 60+ futures, explained by gamma hedging) | Peer-reviewed but on futures; **ours:** last-half-hour SPY-style rule loses to costs on stocks (-3.8 Sharpe) | Shorting in the single-ticker engine; could test on SPY/QQQ only | low |
| S13 | Small items left from the first plan: ORB 10R target and "stocks in play", overnight day-of-week/earnings filters, pre-holiday effect | ORB papers are SSRN working papers; **ours:** ORB and overnight already fail after costs (gross edges too small) | Minute data we have; earnings filter needs D3 | low |
| S14 | Pretrained time-series models (Chronos, TimesFM); LSTM next-day direction | Evidence against: lose to CatBoost zero-shot (Rahimikia et al. 2025); LSTM direction has no profit after costs since 2010 (Fischer & Krauss, *EJOR* 2018) | | skip |

### P3. Method improvements (cheap, do alongside)

- M1. Report alpha against Fama-French 5 factors + momentum (D2), not only against
  equal-weight buy and hold.
- M2. Count every configuration ever tried (across commits) for the luck bar, not only
  the current registry.
- M3. Turnover and cost per strategy in the summary (NMV's 50% monthly turnover line).

## Recommended order

1. D2 (an afternoon, free) and M1: tells us whether current "alphas" are factor exposure.
2. D1 (decide free vs paid with sergio), then rerun all daily experiments on the long
   history: the biggest gain in statistical power.
3. S1 buy/hold spread and S2 multi-horizon trend (cheap, on the strongest evidence).
4. D3, then S3 (value/profitability/investment) and S7 (PEAD).
5. S4 seasonality and S5 factor momentum (need D1/D2).
6. D4, then S8 (deep stat arb) and S9 (chart CNN) on the VPS.
7. S10 to S13 only if earlier results justify them.

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
