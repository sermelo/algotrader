"""Ticker universes used for experiments.

CORE: index ETFs plus large caps across sectors, plus the three original tickers.
Chosen in 2026 from today's large caps, so it carries survivorship bias: compare
strategies with buy-and-hold on the same tickers, not with zero.

BROAD: CORE plus sector, bond and gold ETFs and a seeded random sample of the
provider's symbols (`random.Random(2026).sample` of the 1,215 symbols not already
listed, sorted list from https://api.hfdatalibrary.com/v1/symbols on 2026-10-16),
filtered by rules fixed before any backtest and using only March 2022 data:
first bar before 2022-04-01, first close >= $5, median >= 100 traded minutes per
day in that month (IEX liquidity). 69 of the 100 sampled and all 11 added ETFs
pass. The provider also lists tickers that later stopped trading, and seven kept
names end early (acquired or renamed: ABC, DISCA, KRTX, SQ, SQSP, SWI, XM); a
rename is treated as an exit. Most leveraged and inverse ETFs in the sample failed
the liquidity rule; SDS (2x inverse S&P 500) passed and is kept.
"""
ETFS = ["SPY", "QQQ", "IWM", "DIA"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "JPM", "BAC", "XOM", "CVX",
          "JNJ", "PFE", "PG", "KO", "WMT", "HD", "INTC", "CSCO", "DIS", "NFLX"]
ORIGINAL = ["ADM", "AMC", "AMGN"]
CORE = ETFS + STOCKS + ORIGINAL

SECTOR_ETFS = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "GLD", "TLT"]
SAMPLE = ["ABC", "ADT", "AFL", "ANET", "ANF", "BHF", "CEG", "CHTR", "CNP", "COIN", "CSGP",
          "CVNA", "DGX", "DISCA", "DLR", "DT", "HPQ", "HRI", "INCY", "ISRG", "ITB", "KDP",
          "KEX", "KMX", "KRTX", "LUV", "LW", "MAT", "NDSN", "NOW", "OKE", "ORLY", "PEP",
          "PRU", "PSA", "PSTG", "RCL", "REGN", "RMBS", "RTX", "SBAC", "SDS", "SEDG", "SHOP",
          "SLG", "SMH", "SPG", "SPOT", "SQ", "SQSP", "SWI", "TECH", "TIP", "TJX", "TLRY",
          "TM", "TOST", "TSM", "TSN", "TTWO", "UNH", "VRSN", "VVV", "WAT", "WHR", "XBI",
          "XM", "YUMC", "ZBH"]
BROAD = CORE + SECTOR_ETFS + SAMPLE

UNIVERSES = {"core": CORE, "broad": BROAD}
UNIVERSE = CORE                      # default
