# Strategy Specification: SMA 20/50 Crossover with Enhanced Risk Controls

## Overview
This document specifies the final rule‑set for the SMA 20/50 crossover strategy developed during the backtest enhancement project. The strategy aims to capture medium‑term trends in US equities while controlling risk through adaptive position sizing, regime filters, and realistic transaction costs.

## Core Rule
- **Universe**: Liquid US equities (current implementation: 50 tickers from IEX minute data, Mar 2022‑present).
- **Signal**: 
  - Compute 20‑period simple moving average (SMA20) and 50‑period simple moving average (SMA50) on closing prices.
  - Enter long when SMA20 crosses above SMA50.
  - Exit long when SMA20 crosses below SMA50.
- **Execution**: Signals are acted on the next bar's open price (1‑bar latency) to avoid look‑ahead bias.

## Risk Controls
1. **Adaptive Position Sizing**
   - Risk per trade: 1% of equity (configurable).
   - Position size limited by volatility: dollar risk per share = ATR(14) × 2.0 (or fixed 1.5% stop if ATR unavailable).
   - Max position fraction: 10% of equity per trade.
   - Result: Volatility‑scaled shares = min( equity × risk% / (ATR × mult), equity × max% / price ).

2. **Regime/Trend Filter (Optional)**
   - Only take longs when price > SMA100 **or** ADX(14) > 20.
   - Designed to reduce losses in sideways/choppy markets.

3. **Stop‑Loss**
   - Primary: ATR‑based stop (ATR × 2.0) below entry price.
   - Fallback: Fixed 1.5% stop‑loss if ATR calculation fails.

## Transaction Cost Model
- **Commission**: $0.005 per share, with a minimum of $1.00 per trade.
- **Slippage**: Proportional to spread: slippage = 0.1 × (High‑Low)/Close × price.
  - Added to entry price, subtracted from exit price.
- **Latency**: Fixed 1‑bar delay (next bar open) plus optional jitter (0‑2 bars random) – implemented in enhanced versions.
- **Partial Fill**: Trade size limited to 10% of bar volume to avoid market impact.

## Performance Expectations (Based on Enhanced Backtest)
- **Annual Return**: ~0.5‑2.0% (varies by ticker and regime filter).
- **Sharpe Ratio**: Typically 0.2‑0.5 for individual tickers; portfolio‑level Sharpe ~0.3‑0.6 after diversification.
- **Max Drawdown**: Usually <5% for diversified portfolio; single‑ticker drawdowns can exceed 20% in volatile stocks.
- **Turnover**: Approximately 6‑8 trades per ticker per year (depends on signal frequency and filters).
- **Note**: Results are pre‑tax and exclude slippage beyond spread model; real‑world performance may be lower.

## Implementation Files
- Core backtest: `/home/hermes/.hermes/scripts/backtest/sma_backtest.py` (original)
- Enhanced version with transaction costs: `/home/hermes/.hermes/scripts/backtest/sma_backtest_txn.py`
- Analytics scripts: portfolio construction (`portfolio_analytics.csv`), visual reports (`visual_reports.py`)
- Pipeline automation: `/home/hermes/run_pipeline.sh`

## Usage
1. Ensure data is present in `$HERMES_CACHE/scratch/tickers/` as `_clean.parquet` files.
2. Run the pipeline: `bash /home/hermes/run_pipeline.sh` (executes analytics, visuals, and git tagging).
3. Outputs:
   - Equity curves and statistics in `$HERMES_CACHE/scratch/results/`
   - Plots in `$HERMES_CACHE/scratch/output/`
   - Git tags for reproducibility.

## Limitations
- Based on 1‑minute data; higher‑frequency effects (e.g., intra‑bar slippage) are approximated.
- Regime filter parameters (SMA100, ADX threshold) are not optimized; further research recommended.
- Strategy is long‑only; short‑selling or inverse positions not considered.
- Performance assumes ability to trade at the modeled prices; liquidity constraints may affect larger capital.

## Future Work
- Walk‑forward optimization of all parameters (SMA lengths, risk%, stop‑loss, regime thresholds).
- Incorporate machine‑learning filters to improve signal quality.
- Test on extended universes including delisted tickers.
- Examine execution algorithms (VWAP, TWAP) to reduce market impact.
