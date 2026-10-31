# Deep Reinforcement Learning Approach for Trading Automation in the Stock Market

**Authors:** Taylan Kabbani, Ekrem Duman  
**Affiliation:** Department of Industrial Engineering, Özyeğin University, Istanbul  
**Journal:** *IEEE Access* 10 (2022) 93564-93574, [doi:10.1109/ACCESS.2022.3203697](https://doi.org/10.1109/ACCESS.2022.3203697)  
**Date:** published 5 Sep 2022  
**Full text:** [papers/access2022-Kabbani-DRL-Trading-Automation-Stock-Market.md](papers/access2022-Kabbani-DRL-Trading-Automation-Stock-Market.md)

## Abstract (paraphrased)
Stock trading is cast as a partially observed Markov decision process (cash, shares held, close
prices, ten technical indicators and a news-headline sentiment score per stock) and solved with
TD3, an actor-critic deep reinforcement learning algorithm with a continuous action (shares to buy
or sell per stock). The authors report a Sharpe ratio of 2.68 on unseen test data and conclude that
deep RL is superior to other machine learning for trading.

## Method
- **State:** cash, shares of each stock, close, sentiment (-1/0/+1 from FinBERT on news headlines
  that mention the stock that day) and ten indicators (RSI, SMA, EMA, stochastic %K, MACD,
  accumulation/distribution, OBV, rate of change, Williams %R, disparity index), look-back 20 days.
- **Action:** per stock a number in [-1, 1] scaled to at most `Kmax` shares bought or sold; sells
  executed first, buys cut to the cash left, no shorting.
- **Reward:** change in portfolio value after a percentage commission (rate not given in the text).
  Trades are priced at the close of the decision day (the commission uses the previous close),
  with zero slippage and zero market impact.
- **Experiment 1:** QCOM and MSFT, daily, 2011-2016, $10,000, 5 seeds x 200 episodes; evaluated
  on the **training data only**. Sharpe 1.43 (close only), 2.75 (+ indicators), 3.14
  (+ sentiment), against 0.85 / 1.4 / 2.4 in a Stanford student working paper (Kaur 2017).
- **Experiment 2:** ten large caps (AAPL MSFT QCOM IBM RTX PG GS NKE DIS AXP), daily 2010-2018,
  $100,000, test 2016-2018 (the exact split is shown only in a figure). Batch-normalised observations and
  normalised rewards were added because the model "had to" have them to generalise.

## Results (as reported)
- Test period: Sharpe **2.68**, profit $110,308 on $100,000.
- The same agent with selling disabled (it buys once and holds): Sharpe 2.00, profit $66,949.
  The authors read the difference as the agent's skill.

## Caveats
- The agent **keeps learning on the test set**, and the regularisation was added after seeing
  that the model did not generalise: the test is not a clean out-of-sample test.
- One test window of about two years, in a strong bull market for these ten stocks; no buy and
  hold of the ten stocks at equal weight, no index benchmark, no confidence interval, one run.
- Fills at the signal day's own close (look-ahead in execution), no spread or slippage.
- Ten stocks chosen in 2022 that did well (survivorship), sentiment from Kaggle headline dumps.
- Experiment 1's Sharpe ratios are in-sample.

## Relevance to algotrader
Low. Its edge over its own buy-and-hold variant is +0.68 Sharpe over two years, inside the
noise, and the evaluation lets the model adapt on the test data. Our own learned models
(gradient boosting, cross-sectional ranking, deep arbitrage, chart CNN) found nothing after costs
with next-open fills; ROADMAP S16 records why a deep-RL agent is not being tested.
