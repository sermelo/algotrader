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
   - The data is 1‑minute OHLCV bars built from IEX trade reports (Mar 2022‑present), from the **HF Data Library** (https://hfdatalibrary.com, maintained by Ahmed Elkassabgi). Free for academic research, CC‑BY‑4.0. Not for trading.
   - The Hugging Face repo `elkassabgi/hfdatalibrary` is documentation only; the earlier `load_dataset('IEEx/minute-bar')` instruction did not work (that dataset does not exist).
   - Register for a free API key at https://hfdatalibrary.com (name, institution, email, role, country). Keys expire every 30 days. **Never commit the key.** Keep it outside the repo, e.g. in `~/.config/hfdatalibrary/token` (chmod 600).
   - Download one ticker (full history, Parquet, "clean" version):
     ```bash
     mkdir -p data/scratch/tickers
     curl -H "X-API-Key: $(cat ~/.config/hfdatalibrary/token)" \
       -o data/scratch/tickers/AAPL_clean.parquet \
       "https://api.hfdatalibrary.com/v1/bars/AAPL?version=clean"
     ```
   - The list of available tickers (about 1,250) is public: `https://api.hfdatalibrary.com/v1/symbols`. Rate limit is 100 downloads per minute.
   - Files have columns `datetime, Open, High, Low, Close, Volume, source` (capitalised, `datetime` as a column, not the index). This is the format the scripts expect.
   - `data/` and `.venv/` are git‑ignored. Run scripts with `HERMES_CACHE=$PWD/data` so inputs and outputs stay inside `data/scratch/`.

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

> **Status: NOT reproduced.** The published CSVs in `results/` do not match what the code in this repo produces, and should be treated as unverified.

Replication attempt (2026‑10‑16): downloaded the clean 1‑minute data for the 3 tickers in `results/summary_with_commission_all.csv` (ADM, AMC, AMGN) and ran `backtest/sma_backtest_voladj.py` unchanged (`HERMES_CACHE=$PWD/data`, about 2 minutes).

| Ticker | Published return / trades | Reproduced return / trades |
|---|---|---|
| ADM | +0.14% / 2 | −100.0% / 10,074 |
| AMC | −8.70% / 454 | −98.7% / 9,962 |
| AMGN | +0.72% / 2 | −99.99% / 10,332 |

Findings:
- The script loses essentially all capital on every ticker (win rate about 5‑9%, about 10,000 trades each), because the entry signal (`SMA20 > SMA50` and a regime filter) flips constantly on 1‑minute bars and costs eat the equity.
- `results/summary_with_commission_all.csv` has **3 tickers, not 50**. Two of them show only 2 trades, which is not plausible for this script.
- The README previously said the published results used "no slippage"; the script actually applies a fixed $0.001/share slippage plus $0.005/share commission (min $1).
- The published results therefore came from a different script, parameters or data than anything in this repo. The generating pipeline (`/home/hermes/run_pipeline.sh`) and the 50‑ticker list are not in the repo.
- The performance numbers in `STRATEGY.md` are likewise unverified.

Next step: a clean, tested reimplementation, then re‑running on the data to get results we can trust.

## License

This project is provided for educational and research purposes. See individual files for any specific notices.