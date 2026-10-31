# Artificial Intelligence Applied to Stock Market Trading: A Review

**Authors:** Fernando G. D. C. Ferreira, Amir H. Gandomi, Rodrigo T. N. Cardoso  
**Affiliation:** CEFET-MG (Brazil), University of Technology Sydney  
**Journal:** *IEEE Access* 9 (2021) 30898-30917, [doi:10.1109/ACCESS.2021.3058133](https://doi.org/10.1109/ACCESS.2021.3058133)  
**Date:** published 9 Feb 2021  
**Full text:** [papers/access2021-Ferreira-AI-Stock-Market-Trading-Review.md](papers/access2021-Ferreira-AI-Stock-Market-Trading-Review.md)

## Abstract (paraphrased)
A systematic review of 2,326 Scopus papers (1995-2019) on AI for stock investing, split into
portfolio optimisation, price prediction, financial sentiment analysis and combinations of these.
For each area it describes the most cited papers per year and recent state-of-the-art work.

## Content
- **Portfolio optimisation:** mean-variance and its extensions (cardinality, transaction cost and
  turnover constraints, CVaR), solved more and more with multi-objective metaheuristics
  (genetic algorithms, NSGA-II and similar).
- **Prediction:** neural networks, SVMs, fuzzy systems, then deep learning (LSTM, CNN) on technical
  and fundamental features; most papers report forecast accuracy or error, not trading results.
- **Sentiment:** first news, then social media (Twitter, StockTwits); classifiers around 80% accuracy.
- **Combinations** (e.g. sentiment + prediction + portfolio) are said to beat single methods, but
  there are few such papers.
- **Gaps named by the authors:** few comparisons of complex models with simple ones, little
  analysis of which features matter, few papers using news and social media together.

## Caveats
- Papers are selected by citation count, so it describes what is popular, not what works.
- It does not assess costs, out-of-sample design, look-ahead or multiple testing in the papers it
  reviews; most of their claims are accuracy figures.

## Relevance to algotrader
Background only; no specific strategy to test. Its gap list (simple baselines, feature
attribution) is what our experiments already do.
