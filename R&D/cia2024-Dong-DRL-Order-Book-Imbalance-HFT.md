# Deep Reinforcement Learning for Optimizing Order Book Imbalance-Based High-Frequency Trading Strategies

**Authors:** Boyang Dong, Daiyang Zhang, Jing Xin  
**Affiliation:** University of Chicago, Georgetown University, UW Madison (master's programmes)  
**Journal:** DOI [10.63575/CIA.2024.20204](https://doi.org/10.63575/CIA.2024.20204) (journal name not printed on the PDF; "Open Access")  
**Date:** PDF created April 2025  
**Full text:** not stored here (the PDF states no licence).

## Abstract (paraphrased)
A Deep Q-Network (buy / sell / hold) trades AAPL on one-second order-book snapshots, with order
book imbalance over the top five levels, spread, best-level volumes, a 10-step price trend and a
5-minute volatility as the state. Over twelve months of 2022 data it reports a 15.2% cumulative
return and Sharpe 1.8, ahead of a moving-average crossover, a market-making rule and random trading.

## Method
- **Data:** AAPL Nasdaq order book, top 10 levels, sampled every second, 1 Jan to 31 Dec 2022.
- **Model:** DQN, replay buffer, batch 64, epsilon-greedy decaying over 50 of 200 episodes, each
  episode 10,000 steps (about one trading day); hyperparameters and architecture by grid search.
- **Benchmarks:** SMA 50/200 crossover, a market maker quoting both sides, random actions.

## Results (as reported)
| Strategy | Cumulative return | Sharpe | Max drawdown |
|---|---|---|---|
| DQN | 15.2% | 1.8 | 5.3% |
| Market making | 10.3% | 1.4 | 6.5% |
| MA crossover | 8.5% | 1.2 | 7.8% |
| Random | -2.1% | -0.3 | 12.4% |

## Caveats
- **Frictionless:** no transaction costs, spread, slippage or latency (the paper says so); for a
  one-second strategy these decide everything.
- No train/test split is described: the reported year appears to be the same data the agent was
  trained and grid-searched on.
- Signs of a generated, unedited text: a leftover writing instruction ("count for this section
  is approximately 500 words, adhering to the requirement of splitting the 1000-word ...") and
  unresolved cross-references ("Error! Reference source not found."). One stock, one year.

## Relevance to algotrader
None. We have no order-book data (IEX 1-minute bars only), and the result is not credible as
reported.
