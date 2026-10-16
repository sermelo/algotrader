# AlgoTrader Repository

This repository contains the code, backtest results, and documentation for the SMA 20/50 crossover trading strategy with enhanced risk controls and transaction cost modeling.

## Directory Structure

```
.
├── backtest/                 # Python backtest scripts
│   ├── sma_backtest.py                   # Original SMA backtest
│   ├── sma_backtest_5min.py              # 5‑minute variant
│   ├── sma_backtest_enhanced.py          # Enhanced with extra features
│   ├── sma_backtest_fixed.py             # Fixed‑version
│   ├── sma_backtest_ml.py                # ML‑enhanced version
│   ├── sma_backtest_ml_simple.py         # Simple ML version
│   ├── sma_backtest_new.py               # New iteration
│   ├── sma_backtest_quick.py             # Quick test version
│   ├── sma_backtest_slippage_latency.py  # Includes slippage/latency
│   ├── sma_backtest_subset.py            # Subset testing
│   ├── sma_backtest_test.py              # Test harness
│   ├── sma_backtest_txn.py               # Transaction cost model
│   ├── sma_backtest_txn_debug.py         # Debug version of txn
│   ├── sma_backtest_voladj.py            # Volatility‑adjusted position sizing
│   ├── sma_backtest_voladj_50_200.py     # Vol‑adj with 50/200 SMA
│   ├── sma_backtest_voladj_atr4.py       # Vol‑adj using ATR(4)
│   ├── sma_walkforward_bar.py            # Walk‑forward bar‑by‑bar
│   ├── sma_walkforward_bar_fast.py       # Faster walk‑forward
│   ├── sma_walkforward_demo.py           # Demo of walk‑forward
│   ├── visual_reports.py                 # Generate equity/plots
│   └── walkforward.py                    # Walk‑over utilities
├── results/                  # CSV outputs from the full backtest
│   ├── summary_with_commission_all.csv   # Per‑ticker stats (return, Sharpe, max‑DD, trades)
│   └── daily_return_buckets_all.csv      # Histogram of daily returns (1% buckets)
├── test_voladj.py            # Driver: runs first 5 tickers, prints summary
├── test_voladj_50_200.py     # Driver for vol‑adj 50/200 SMA
├── test_voladj_atr4.py       # Driver for vol‑adj using ATR(4)
├── STRATEGY.md               # Full strategy specification
└── README.md                 # This file
```

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/sermelo/algotrader.git
   cd algotrader
   ```

2. **Environment** (if not already set up via Hermes)
   - Python 3.11+
   - Required packages: `pandas`, `numpy` (others may be imported by specific scripts)
   - The scripts expect minute‑OHLCV data in Parquet format under `$HERMES_CACHE/scratch/tickers/` (or adjust the path in the scripts).
   - Commissions modeled: IBKR per‑share $0.005, minimum $1.00 per trade (as used in the backtest).
   - Platform: Linux, Hermes agent environment (scripts tested on Hermes).

3. **Data Acquisition** (if data not already present)
   - The backtest uses 1‑minute OHLCV data from the IEX feed via the Hugging Face Data Library (Mar 2022‑present).
   - To download, you can use the `datasets` library:
     ```bash
     pip install datasets
     python -c "from datasets import load_dataset; ds = load_dataset('IEEx/minute-bar', split='train'); ds.to_parquet('$HERMES_CACHE/scratch/tickers/')"
     ```
   - Alternatively, you can obtain the data from Kaggle (e.g., `StockMinuteData`) or from a paid provider (IQFeed, FirstRate, TickData) and convert to Parquet files named `<TICKER>_clean.parquet`.
   - Ensure each file contains columns: `open, high, low, close, volume` with a datetime index.

4. **Run a quick test**
   ```bash
   python test_voladj.py
   ```
   This processes the first five tickers and prints Sharpe, total return, and trade counts.

5. **Run the full backtest** (as used to generate the CSVs in `results/`)
   - The original pipeline executed the suite of scripts and produced the CSVs.
   - To regenerate, run the appropriate backtest script (e.g., `backtest/sma_backtest_voladj.py`) over the full ticker list and aggregate the outputs.
   - Example (pseudo):
     ```bash
     python backtest/sma_backtest_voladj.py --tickers-file path/to/ticker_list.txt
     ```
   - After generating per‑ticker CSVs, you can combine them into `summary_with_commission_all.csv` and compute the daily‑return histogram.

6. **Refer to the strategy doc**
   See `STRATEGY.md` for detailed rules, risk controls, transaction cost model, performance expectations, usage instructions, limitations, and future work.

## Reproducing the Published Results

The CSV files in `results/` were generated from a backtest of 50 US equities (IEX minute data, Mar 2022‑present) using an SMA 20/50 crossover with:

- **Volatility‑adjusted position sizing**: 1% risk per trade, ATR‑based stop (ATR × 2.0), max position 10% of equity.
- **IBKR per‑share commission**: $0.005/share, minimum $1.00 per trade.
- **No additional slippage or latency** beyond the commission model (the `*_txn.py` scripts model slippage/latency, but the published results used the commission‑only version).

To regenerate exactly:

1. Ensure the 50 ticker `_clean.parquet` files are present in `$HERMES_CACHE/scratch/tickers/`.
2. Run `backtest/sma_backtest_voladj.py` (or the version used in the pipeline) over all tickers.
3. Collect the per‑ticker statistics (the script already writes a summary CSV; otherwise aggregate manually).
4. Compute the daily‑return histogram from the equity curves if needed (the `daily_return_buckets_all.csv` was produced by the pipeline’s aggregation step).

## License

This project is provided for educational and research purposes. See individual files for any specific notices.