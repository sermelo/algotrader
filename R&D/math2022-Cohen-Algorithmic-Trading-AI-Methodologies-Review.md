# Algorithmic Trading and Financial Forecasting Using Advanced Artificial Intelligence Methodologies

**Author:** Gil Cohen  
**Affiliation:** Department of Management, Western Galilee Academic College, Israel  
**Journal:** *Mathematics* 10(18) (2022) 3302 (MDPI), [doi:10.3390/math10183302](https://doi.org/10.3390/math10183302)  
**Date:** published 12 Sep 2022  
**Full text:** [papers/math2022-Cohen-Algorithmic-Trading-AI-Methodologies-Review.md](papers/math2022-Cohen-Algorithmic-Trading-AI-Methodologies-Review.md)

## Abstract (paraphrased)
A narrative review of recent papers that use machine learning and deep learning (SVM, LSTM, fuzzy
systems), social-media sentiment and chart-pattern recognition to forecast and trade stocks, ETFs,
currencies, cryptocurrencies and precious metals. It concludes that most reviewed systems trade
successfully and beat buy and hold.

## Content
- **SVM:** direction accuracy of about 55-65% on daily or minute crypto data; hybrids with ARIMA.
- **LSTM:** e.g. Ghosh, Neufeld & Sahoo (2021): LSTM / random forest on S&P 500 stocks
  1993-2018, long the 10 and short the 10 stocks most likely to beat the market each day, 0.64% a
  day (the review does not say whether after costs).
- **Fuzzy systems:** rule-based systems on technical indicators, several claims of beating buy
  and hold (Athens exchange, Bitcoin, precious-metal futures with RSI on 60-minute bars).
- **Sentiment:** Twitter / StockTwits / Google Trends, mostly for cryptocurrencies; one claim of
  89.9% direction accuracy on the SSE50 index.
- **Patterns:** candlesticks ("engulfing" on Bitcoin, profit factor 3.54), Bollinger bands,
  Darvas boxes.

## Caveats
- No selection method beyond "originality and importance"; no critical reading of the papers
  (costs, out-of-sample design, data snooping), so "most papers proved successful" reflects
  publication bias. Many results are cryptocurrency or accuracy-only.
- Several claims are the reviewed papers' in-sample or pre-cost numbers.

## Relevance to algotrader
Low. The one concrete equity result it cites (daily LSTM / random forest long-short on the
S&P 500) extends Fischer & Krauss (2018), already rejected in ROADMAP S14: there the profits
vanish after costs from about 2010.
