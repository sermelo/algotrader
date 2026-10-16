# Research & Development (R&D) – Intraday Trading Papers

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

## Full text

`papers/` holds the full text of each paper in Markdown, under the same file name as its
summary. Each starts with its source and Creative Commons licence (all five are CC; see
the header of each file). Four were converted from the arXiv HTML version; 2405.13959 has
no HTML version and was converted from the PDF without figures. Some equations and
tables may be garbled by conversion: check the arXiv original for anything that matters.
