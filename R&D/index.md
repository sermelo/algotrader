# Research & Development (R&D) – Trading Papers

Summaries of selected academic papers on intraday trading strategies, forecasting and risk
management. Each summary gives the paper's title, authors, arXiv link, a paraphrased abstract, the
method and results as the paper reports them, caveats, and relevance to this backtester.

Numbers below are the papers' own claims, not reproduced by this code.

## Papers (most relevant first)

1. **[2405.13959] Decision Trees for Intuitive Intraday Trading Strategies**  
   Per-stock decision trees on 9 technical features (RSI, ADX, SMA/VWAP ratios, volatility) with
   1-minute NIFTY50 bars. On the test period, 46% of stocks beat buy-and-hold on return and 60% on
   Sharpe. No commissions or slippage. **Tested here** (`DecisionTreeStrategy`): a gross edge of
   0.15 to 0.35 bp per trade on 1-minute bars, wiped out by commissions (-68% to -99%).

2. **[2311.18477v3] Intraday FX Volatility-Curve Forecasting with Functional GARCH Approaches**  
   Functional GARCH forecasts of intraday volatility curves and VaR for USD/EUR, USD/GBP, USD/JPY,
   using multi-level FPCA and bid-ask spread curves. Relevant only as a possible risk/stop input.

3. **[2411.10956v2] IVE: Enhanced Probabilistic Forecasting of Intraday Volume Ratio with Transformers**  
   Transformer forecasting the mean and standard deviation of 1-minute volume ratios for VWAP
   execution; about 2.5 months of live trading in Korea. Execution, not signals.

4. **[2103.13507v1] Intraday trading strategy based on time series and machine learning for Chinese stock market**  
   Markowitz portfolio plus an MLP for intraday trades under China's T+1 rule. The quoted "18.24%
   yearly return" is one month (October 2020, +1.41%) annualised. Too little evidence.

5. **[2402.01215v2] Intraday Power Trading for Imbalance Markets: An Adaptive Risk-Averse Strategy using Mixture Models**  
   Mixture-model imbalance price forecasts and risk-averse optimisation with an online-tuned risk
   parameter, on the Belgian electricity market. Not applicable to equities.

## AI reviews and deep reinforcement learning (added 2026-10-31)

Uploaded by sergio. None gives a strategy worth testing here.

6. **[Kabbani & Duman, IEEE Access 2022] Deep Reinforcement Learning Approach for Trading Automation in the Stock Market**  
   TD3 agent on ten large caps (daily, technical indicators plus FinBERT headline sentiment),
   Sharpe 2.68 on 2016-2018 against 2.00 for its own buy-once variant. The agent keeps learning on
   the test set, fills at the signal day's close, no spread. Not tested (ROADMAP S16).

7. **[Dong, Zhang & Xin, 2024/25] Deep Reinforcement Learning for Optimizing Order Book Imbalance-Based HFT**  
   DQN on one-second AAPL order-book imbalance, 15.2% and Sharpe 1.8 over 2022, with no costs and
   no stated train/test split; the text has unedited generation leftovers. We have no order-book
   data. No licence stated, so no full text here.

8. **[Ferreira, Gandomi & Cardoso, IEEE Access 2021] Artificial Intelligence Applied to Stock Market Trading: A Review**  
   Bibliometric review of 2,326 Scopus papers (portfolio optimisation, prediction, sentiment,
   combinations), chosen by citations; reports accuracy claims, does not assess costs or
   out-of-sample design. Background only.

9. **[Cohen, Mathematics 2022] Algorithmic Trading and Financial Forecasting Using Advanced AI Methodologies**  
   Narrative review (SVM, LSTM, fuzzy systems, sentiment, chart patterns), many crypto and
   accuracy-only results, no critical reading. Its one equity result (daily LSTM long-short on
   the S&P 500) is the Fischer & Krauss line already rejected (ROADMAP S14).

A fifth upload, Shi et al., "Market incentives, carbon quota allocation and carbon emission
reduction" (*Journal of Environmental Management* 319, 2022), is about China's carbon-emission
trading pilots, not securities trading; it is not kept here.

## Full text

`papers/` holds the full text of each paper in Markdown, under the same file name as its
summary. Each starts with its source and Creative Commons licence (see the header of each
file). Four were converted from the arXiv HTML version; 2405.13959 has no HTML version and
was converted from the PDF without figures, as were the three 2026-10-31 papers (the Dong et
al. paper has no licence and is not included). Some equations and
tables may be garbled by conversion: check the original for anything that matters.
