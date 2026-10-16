# algotrader

Backtesting of an SMA 20/50 crossover strategy (long-only, US equities, 1-minute IEX data) with volatility-adjusted sizing and commission/slippage models. Goal: test different algorithmic trading strategies.

@README.md
@STRATEGY.md

## Notes for Claude

- The code was written by a non-programmer agent (Hermes) and has not been reviewed. Treat it as a prototype, not trusted code.
- `backtest/` holds many near-duplicate script variants (`_fixed`, `_new`, `_test`, `_quick`, `_txn_debug`, ...) rather than one parameterised module. Prefer consolidating over adding another copy.
- Scripts and docs reference Hermes-specific paths (`/home/hermes/...`, `$HERMES_CACHE/scratch/tickers/*_clean.parquet`). The input data is not in this repo, so nothing runs here until a data path is set up.
- Published results (`results/`) used the commission-only vol-adjusted version; the slippage/latency model was not part of them.
- Verify claims in STRATEGY.md (performance numbers, regime filter, partial fills) against the code before relying on them.
