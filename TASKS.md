# Tasks

Open work items. Keep it current: tick items when done (with the commit), add new ones as they come up. Longer-term
research priorities are in `ROADMAP.md`; `HANDOFF.md` has VPS access and lessons. Last updated 2026-10-28 20:00 UTC.

## State right now (2026-10-28 20:00 UTC)

- **Everything committed and pushed** (fundamentals fix + rerun results, paper.py rate fix, extremes step 1, docs).
  The VPS's `~/algotrader-work` gets this commit's code (rsync) after it; scratch folders `~/algotrader-extremes`
  and `data/tmp_check` deleted. No run is going on the VPS.
- **Paper account** DUR288542 (IBKR paper, base EUR), IB Gateway running on the VPS (started by hand 2026-10-27; IBC
  restarts it daily at 06:00 UTC; IBKR asks sergio for a 2FA approval on his phone about once a week). Cash
  **$14,435.11 + EUR 48.21** after the conversion (settles 2026-10-29), no positions, no open orders, no
  `data/paper/state.json` (so the first `daily --submit` run builds the portfolio at the next open). No crontab entries.
  `paper status` works ($14,662 shown with the ECB rate; $14,612.50 with IBKR's after the 3b fix).
- **Live account** (sergio's own): EUR 1,000, leave it in euros until live trading starts (then convert once per
  deposit; each conversion costs at least ~$2).

## Paper trading: two books (2026-10-28/29)

- [x] sergio chose **momentum top 5 weekly** as his candidate for real money (keep recommending it with its caveats;
  memory `momentum-top5-weekly-candidate`) and wants it paper-traded next to operating profitability. IBKR gives one
  paper account, so `paper.py` runs **books**: op profitability top 10 monthly and momentum top 5 weekly, $15,000 each,
  orders netted per stock, fills split back (`allocate`), books must add up to the account. Momentum prices: IBKR
  `ADJUSTED_LAST` daily history, 2 years, ~0.6 s a stock (checked 2026-10-29: 501 bars for AAPL, BRK B, MA, NVDA).
- [x] Paper balance reset by sergio to EUR 30,109 (about $33,559) on 2026-10-29. The Gateway failed to log in again
  after the reset (2026-10-28 20:34 UTC) and stopped at the 06:00 restart; restarted 2026-10-29 11:38 UTC, credentials
  unchanged.
- [x] **EUR -> USD again** (done 2026-10-29 ~12:05 UTC, sergio's OK): sold EUR 27,000 at 1.11459 (limit 1.1021, 1% under
  the ECB's 1.1133; commission EUR 1.79), about $30,094; about EUR 3,100 left. Account $33,558. Dry run of both books
  (2026-10-28 close, `data/paper_dry/` on the VPS, delete at the start): op CL GDDY LYV VRSK IT TPR MA APP STX AAPL,
  momentum SNDK MRNA LITE MU STX; 14 net orders.
- [ ] Cron (item 5, still postponed until after the Tiingo batch, needs sergio's OK).

## Before paper trading starts

**2026-10-27: start postponed until after the Tiingo batch (item 10) and its reruns (sergio).**

Target: first decision after the close on **Friday 2026-10-30** (last NYSE session of October), filled with
market-on-open orders at the **Monday 2026-11-02** open.

1. [x] **Stale SEC values in `fundamentals.py` (bug): code and tests done 2026-10-27, uncommitted (see item 2).**
   Found in the pre-start review: `_known` chose one XBRL tag per company for its whole history, so a company that
   switched tags kept its last value under the old one (Qualcomm: FY2019 `StockholdersEquity` $4.9B against FY2025
   operating income $12.4B, score 2.52 instead of 0.58); values were carried forward forever (H&R Block 35 months in
   the top 10 on two-year-old income); a ratio could mix parts from different years. Fix: tag chosen per period
   (`_known`), ratio parts from the same period end (`_combine`, as-of values of each part), asset growth only over
   consecutive fiscal years (`_previous_year`, 330-400 days), values expire 18 months after their period end
   (`MAX_AGE`, `_latest_panel`; Fama-French convention). Documented in STRATEGY.md ("Fundamental sorts") and README.
   Measured on the backtest facts (month ends 2009-2026, members), old vs new top-10 overlap: operating
   profitability 9.1 of 10, asset growth 9.7, gross profitability 7.1 (old: Amazon 0.01-0.08 instead of ~0.5, negative
   gross profit for ETN, MPC, EPAM), book-to-market 6.6 (old: BRK-B 1,910, FOX 330 million, TRIP 8,688 from stale share
   counts). Live on 2026-10-27 the fix swaps QCOM for AAPL: top 10 = CL, GDDY, LYV, VRSK, IT, TPR, MA, APP, STX, AAPL
   (356 of 503 members scored, 387 before).
2. [x] **Finish the rerun, merge, regenerate, commit** (done 2026-10-28: merged, every fingerprint current, no verdict
   changed, ROADMAP 19; deep arbitrage -5.37 / -4.79 without HF as before) (every experiment whose fingerprint covers `fundamentals.py`:
   all long-history families; not merger, core or broad). When `all.done` exists on the VPS:
   1. `rsync -a -e "ssh -i $K" $H:algotrader-work/data/rerun_f/ data/rerun_f/`
   2. Replace the result folders' experiment files (compare file lists so nothing from the old run is left over):
      `results/long/` <- `rerun_f/long/*` plus `rerun_f/long_deep/deep_arb_long_*` and its manifest entry merged into
      `results/long/manifest.json`; `results/long_nohf/` <- `rerun_f/long_nohf/*` plus `rerun_f/long_nohf_deep/` the same
      way; `results/long_stop/` <- `rerun_f/long_stop/*`; `results/long_conc/`, `results/long_rebal/`, `results/long_vol/`
      <- their `returns.csv` + `manifest.json`; `results/mid/` <- `rerun_f/long_mid/*`.
   3. Summaries: `results/long` and `results/long_nohf` with `longrun.write_summary(dir)` after registering the combo
      and chart experiments (as `longrun_combo.main` / `longrun_chart.main` do, so their stale check works);
      `uv run python -m algotrader.concentration report`, `... rebalance report`, `... volsubset report`; long_stop and
      mid summaries come from the run itself (check they list every experiment).
   4. Reports: `uv run python -m algotrader.french report --results results/long results/mid`,
      `uv run python -m algotrader.drawdown`, `uv run python -m algotrader.trials`,
      `uv run python scripts/strategy_blend.py results/long results/long/blend.md`, `uv run python scripts/rank_configs.py`.
   5. Fingerprints: every entry of `results/long*/manifest.json`, `results/mid/manifest.json` must match the current
      code (summaries mark "(stale)"; none may remain). No new configurations, so the trials count stays 1,226.
   6. Docs with the new numbers: README "Long history" (all bullets: long-only list, chart CNN, deep arb, combo and
      trend filter, blends, stop-losses, concentration incl. "best 1 strategy chosen yearly", weekly, volatile subset),
      "Mid-cap test" (should be unchanged: price-only), STRATEGY.md result lines (concentration "2 of 11" unchanged),
      `docs/paper-trading.md` (backtest numbers 0.99 / +16.7% / -41.0%, today's target), ROADMAP (new item for the
      fix), HANDOFF; then `uv run pytest`, one commit (code + results + docs), push; delete `data/tmp_check` on the VPS.
   Results already known (concentration, `rerun_f/long_conc`, done 14:38 UTC): paper configuration (operating
   profitability top 10) Sharpe 0.98 -> 0.99, CAGR +16.0% -> +16.7% a year (+1.30% a month), FF alpha +3.0% (t 1.30) ->
   +3.3% (t 1.38), max drawdown -34.9% -> **-41.0%** (Jan-Mar 2020; SPY -33.7%), worst month -10.5% -> -17.9%
   (March 2020), turnover 92% -> 90% a year. Top 5 1.04 -> 0.94 (no longer the best Sharpe; the "best 1 strategy
   chosen yearly" row is now 0.94, +17.3%). Since 2019: +15.1% a year against SPY +17.2% (old code +12.3%), FF alpha
   +0.6% (t 0.14); 2012-18 +18.5% against +12.6%, alpha +5.4% (t 2.22). Combined signal (uses gross profitability)
   moved most: top 3 0.17 -> 0.54, top 5 0.21 -> 0.46. Pre-registered verdict unchanged (2 of 11 pass: momentum,
   asset growth). Price-only strategies identical to the previous run (as they must be).
3. [x] **EUR -> USD in the paper account** (done 2026-10-27 14:22 UTC, sergio's OK): sold EUR 12,950 at 1.11468 USD
   (IDEALPRO odd lot, limit 1% under the ECB reference, filled at once; commission EUR 1.79). Cash now $14,435.11 and
   EUR 48.21, settling 2026-10-29. Without it, buying the stocks would have borrowed the dollars at about 4% a year.
   3b. [x] **Runner's dollar value** (done 2026-10-27, uncommitted, in the working tree with item 1): with dollars
   held IBKR reports the rate as `$LEDGER-ExchangeRate` USD (0.8970318 EUR per USD = 1.11479), not `ExchangeRate`;
   `paper.usd_rate` now reads it before the ECB fallback (checked against the live Gateway: $14,612.50 instead of the
   ECB's $14,662). `equity.csv` also records base currency, value in it, rate and source (`ibkr`/`ecb`); an existing
   file in the old format gets the new columns without losing rows (the VPS has one dry-run row). 2 new tests (363).
   **Deploy to the VPS with the item 2 rsync, before Friday's cron;** check `paper status` shows "ibkr".
4. [x] **Failure alerts** (done 2026-10-27, sergio chose option a): Telegram sender copied to the VPS
   (`~/.config/telegram-notify/`, 700/600), `notify_cmd` set in `config/paper.json` (here and on the VPS, uncommitted
   here); test messages sent from the VPS directly and through `paper.notify`. Errors (Gateway down, 2FA pending) arrive
   as "paper <day>: ERROR ...".
5. [ ] **Cron jobs: POSTPONED (sergio, 2026-10-27): paper trading waits until the Tiingo second batch (item 10)
   is in and the long reruns are done.** Not on Friday 2026-10-30. Note told to him: the live signal uses no price
   history (today's members, SEC facts, previous close), so the batch changes the backtest reference numbers, not
   the orders. Start date to agree with him then (as soon as ready, building at once, or at the November month end,
   Monday 2026-11-30). Still needs his explicit OK. Original plan: On the VPS:
   `30 21 * * 1-5 cd ~/algotrader-work && ~/.local/bin/uv run python -m algotrader.paper daily --submit >> data/paper/cron.log 2>&1`
   and an `@reboot` entry for `~/ibkr/start-gateway.sh` (`docs/paper-trading.md` step 7). Install on Friday
   2026-10-30 before 21:30 UTC: with no state the first run builds the portfolio at once, so an earlier start would
   build mid-week and decide again on Friday. Before it: `paper status` and a dry `paper daily` (expect the target
   above, with AAPL and without QCOM; the VPS already runs the fixed code).

## Decisions and advice from the review (2026-10-27)

- **Strategy kept:** operating profitability, top 10, monthly, $15,000 paper. Expect about the market, with drops at
  least as deep as the market's (backtest -41% in 2020). Purpose of paper trading: a few weeks to ~3 months to find
  bugs and check orders, fills, costs and records against the backtest; profit cannot be proven in months (with a
  Sharpe near 1 it takes about 4 years). **Correction:** before the fix I said it trailed SPY by 4.8% a year since 2019;
  with the fix it is 2.1% a year (FF alpha t 0.14).
- **No tuning:** Sharpe is flat across N = 3..50, monthly = weekly = daily, buffers cut turnover but not risk, the
  volatile subset is worse; any tweak picked on this history is try number 1,227.
- **No leverage (1.0x):** half-Kelly with a sober 6-8% a year and ~4% IBKR borrowing is 0.6-0.9x; with the fixed
  backtest the worst drop is -41% at 1x, -58% at 1.5x, -73% at 2x (SPY 2008 at 1.5x: -68%).
- **Capital for live trading** (IBKR Tiered, today's top 10 and prices; scratch calculation, not committed): trading
  costs 0.1-0.2% a year from $10,000 up (Fixed plan about twice that), so commissions are not the limit; whole shares
  are: idle cash 61% at $2,000 (5 names unbuyable), 37% at $5,000 (MA, STX unbuyable), 17% at $10,000, 9% at $15,000,
  6% at $25,000, 3.5% at $50,000. Minimum ~$10,000, comfortable $25,000+; sergio's EUR 1,000 is too small for this
  strategy (fractional shares with market-on-open orders not checked). Bigger than commissions: Spanish tax on gains
  realised every year (90% turnover) against an index fund's deferral and tax-free transfers (rough 0.5-1% a year;
  ask a tax adviser), and whether the strategy beats an index fund at all.
- The strategy does not select volatile stocks (sergio asked); the volatile-subset idea was tested on 2026-10-22 and
  made operating profitability worse.

## Optional, after paper trading has started

6. [ ] **Robustness check of the signal's definition** (pre-register before running): the top 10 by operating
   income / book equity is dominated by near-zero equity after buybacks (Colgate 61x: $3.3B on $54M). Rank by
   operating income / total assets instead and compare; tells whether the past result is profitability or the
   near-zero-equity quirk.
7. [ ] **Volatility-managed overlay** (pre-register; Moreira & Muir 2017, against Cederburg et al. 2020): scale
   exposure by target / realised volatility, capped at 1x, for smaller drawdowns. A rough check on the old daily
   returns (not the engine) gave max drawdown -21% instead of -35% for +14.3% a year instead of +16.0%.
8. [ ] **Untested strategies with published evidence and free data:** shareholder yield (dividends + buybacks -
   issuance; Boudoukh et al. 2007), merger arbitrage entered on the announcement day (8-K press releases; follow-up
   of S15), Lazy Prices (10-K text changes; Cohen, Malloy & Nguyen 2020).
   Added 2026-10-27 (sergio OK, after item 11): **insider buying** from SEC Form 4 open-market purchases (code P),
   free point-in-time history from SEC bulk/EDGAR filings (Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012:
   'opportunistic' vs routine insiders). Pre-register: signal known from the Form 4 acceptance time, monthly top N.
11. [x] **Biggest monthly winners and losers** (sergio, 2026-10-27): each month, review the S&P 500 members with the
   highest and the lowest return over that month (returns, not accounting profit), and look for a rule that would
   have identified them beforehand, to hold the likely winners or avoid the likely losers.
   - **Step 1, descriptive (no trading):** for every month 2009-2026, the top and bottom 10 (and decile) by the month's
     return; what was known at the previous month end: past returns (1, 6, 12 months), volatility, maximum daily
     return, distance from the 52-week high, size, sector, value, profitability, asset growth, an earnings release
     due in the month, recent index entry. For each, the lift: how much more often a stock with it lands in the top or
     bottom group than the average member, by year.
   - **Expected trap:** extreme months come mostly from volatility and news (earnings, deals), so high-volatility
     stocks fill BOTH tails; a predictor of "extreme" is not a predictor of direction, and avoiding the worst usually
     also means missing the best (the stop-loss test found the same). What counts is an asymmetric lift: more
     often in one tail than in the other, stable across years.
   - **Literature to start from:** lottery stocks (Bali, Cakici & Whitelaw 2011, the MAX effect: stocks with extreme
     past daily returns underperform), distress risk (Campbell, Hilscher & Szilagyi 2008), earnings-announcement
     premium (Frazzini & Lamont 2007), short-term reversal and momentum (already tested here).
   - **Step 1 done 2026-10-27** (uncommitted; `algotrader/extremes.py`, `tests/test_extremes.py`,
     `scripts/monthly_extremes.py`, `results/extremes/`, README section "Biggest monthly winners and losers"; run on the VPS
     from a separate tree `~/algotrader-extremes` with `data` symlinked, so the rerun's checkout was not touched; under a
     minute). Result: the expected trap. Volatility, largest/worst daily return, 52-week-high distance, past losers, earnings
     due and new members lift BOTH tails (most volatile fifth 1.80 top decile / 1.76 bottom decile). Largest asymmetries are
     industries (telecom -0.25, energy -0.20, tech +0.11, software +0.10), the most stable is operating profitability Q5
     (+0.09, 15 of 18 years, the paper signal itself). MAX effect not present (Q5 - Q1 +0.43%/month, t 1.26). No Q5 - Q1
     |t| above 1.4. **Step 2 not run** (sergio agreed 2026-10-27: no step 2 without a step-1 finding).
   - **Step 2, only if step 1 finds a stable asymmetric lift:** pre-register one rule (e.g. exclude the flagged
     stocks from the paper strategy's top 10, or hold the flagged winners), walk-forward on the long history, causal
     features only, placebo with shuffled months, configurations counted in `results/trials.md`. Lesson from the
     intraday test: knowing afterwards which stocks were extreme is easy; only a rule a trader could follow at the
     month end counts.

## With the November long rerun (after the Tiingo batch, item 10)

12. [ ] **Broken prices in the long-history data** (found 2026-10-27 by item 11; existed before, affects the long
   backtests too). In the frames `longrun.load_frames` builds: negative adjusted closes in the Hugging Face series of RRD
   (2,204 days), WIN (1,841, before its April 2015 spin-off; WIKI is positive there) and WM (187; probably Washington Mutual
   and Waste Management under one ticker): `hfdata.adjust_frame` turns a large event (spin-off or special dividend bigger
   than the previous close) into a negative factor. About 45 more series have a one-day move under -60% or over +150%:
   some real (SIVB, SBNY, FRC, 2008-09), some unadjusted spin-offs (ITT 2011 -78%, in WIKI as well), some probably reused
   tickers (TKO, WBD, XYZ, STI, MMI). To do: refuse a non-positive adjustment factor in `hfdata` (test), list each jump's
   cause, cut or drop the broken pieces (as `midcap.segments` does), then rerun the long families with item 9/10. Check
   how much the backtests used them (membership months, holdings) first.

9. [ ] **Book-to-market market value still wrong in two cases** (found 2026-10-27 while checking the fix, not caused
   by it): (a) multi-class companies report one class as the non-dimensional share count (SPG's 8,000 Class B
   shares, BRK-B, FOX), so the market value is far too small and book-to-market absurd (SPG 4,417); (b) after a
   split the share count from the last filing is pre-split while the raw close is post-split (CMG 2024, CSX 2021),
   for a few months. Months with book-to-market > 5 among members: BRK-B 29, SPG 15, ANF 9, PSA 6, CRM, LH, FTI,
   DXC 4 (AIG and GNW in 2009-12 are genuine). Fix idea: shares split-adjusted with the adjusted/raw price ratio at
   the share count's date; a plausibility check or the cover page's public float for multi-class companies.
   Affects only `value*` and the concentration/rebalance value rows, not the paper strategy.

## Dated

10. [ ] On or after 2026-11-01: Tiingo second batch (ROADMAP pending 2: `scripts/history_queue.sh` on the VPS for the
   remaining ~216 symbols; 166 current members' prices end 2026-03-31 until then), then `history check` and the long
   reruns (combine with item 9).
