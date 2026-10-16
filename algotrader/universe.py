"""The ticker universe used for experiments.

Index ETFs plus large caps across sectors, plus the three original tickers. Chosen
in 2026 from today's large caps, so it carries survivorship bias: compare strategies
with buy-and-hold on the same tickers, not with zero.
"""
ETFS = ["SPY", "QQQ", "IWM", "DIA"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "JPM", "BAC", "XOM", "CVX",
          "JNJ", "PFE", "PG", "KO", "WMT", "HD", "INTC", "CSCO", "DIS", "NFLX"]
ORIGINAL = ["ADM", "AMC", "AMGN"]
UNIVERSE = ETFS + STOCKS + ORIGINAL
