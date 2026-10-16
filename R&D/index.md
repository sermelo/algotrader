# Research & Development (R&D) – Intraday Trading Papers

This directory contains concise markdown summaries of selected academic papers related to intraday trading strategies, forecasting, and risk management. Each file includes the paper’s title, authors, arXiv link, abstract, and a plain‑language summary.

## Papers

1. **[2103.13507v1] Intraday trading strategy based on time series and machine learning for Chinese stock market**  
   Combines Markowitz portfolio optimization with an MLP to predict intraday price movements under China’s T+1 rule, achieving an annual return of ~18%.

2. **[2402.01215v2] Intraday Power Trading for Imbalance Markets: An Adaptive Risk‑Averse Strategy using Mixture Models**  
   Uses mixture models to forecast imbalance prices and a risk‑averse stochastic optimization with online‑tuned risk parameter, validated on the Belgian electricity market.

3. **[2411.10956v2] IVE: Enhanced Probabilistic Forecasting of Intraday Volume Ratio with Transformers**  
   Introduces a Transformer‑based model (IVE) that predicts the full distribution (mean ± std) of intraday volume ratios, improving VWAP execution and spike detection.

4. **[2405.13959] Decision Trees for Intuitive Intraday Trading Strategies**  
   Learns decision‑tree rules from technical indicators (RSI, MACD, etc.) for NIFTY50 stocks, generating stock‑specific intraday trading rules that beat buy‑and‑hold.

5. **[2311.18477v3] Intraday FX Volatility‑Curve Forecasting with Functional GARCH Approaches**  
   Models intraday return curves as functional observations in a GARCH framework, using FPCA and bid‑ask spreads to forecast volatility and intraday VaR for major FX pairs.

## How to Use
Each markdown file can be read directly or used as a quick reference when designing or backtesting intraday strategies. The papers cover a range of approaches—from traditional econometrics (GARCH) to modern machine learning (Transformers, decision trees)—and various asset classes (equities, power, FX).
