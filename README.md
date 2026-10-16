# algotrader

A small, tested backtester for long-only SMA-crossover strategies on US equity
minute bars. Written to replace an earlier, unverified set of scripts (see
[Removed legacy code](#removed-legacy-code)).

- `algotrader/` is the package, in a handful of small classes (see [Design](#design)).
- `tests/` has the unit tests (`pytest`, 44 tests, about 1 second).
- `STRATEGY.md` is the exact strategy and execution rules.

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # or: pip install -e ".[dev]"
.venv/bin/python -m pytest                                              # run the unit tests
```

Get data (next section), then:

```bash
.venv/bin/python -m algotrader.run --data-dir data/scratch/tickers \
    --tickers ADM AMC AMGN --timeframe 5min --out data/results/summary_5min.csv
```

Omit `--tickers` to run every `*_clean.parquet` in the folder. `--help` lists all
options (`--fast`, `--slow`, `--trend-filter`, `--risk`, `--max-position`,
`--slippage`, `--capital`).

Or from Python:

```python
from algotrader import BarLoader, Backtester, Metrics

bars = BarLoader("data/scratch/tickers").load("AMGN", timeframe="30min")
result = Backtester().run(bars)            # default: SMA 20/50, 1% risk, IBKR commissions
print(Metrics(result, bars).summary())
print(result.trades_frame().head())
```

## Data

The data is 1-minute OHLCV bars built from IEX trade reports (Mar 2022 to
present), from the **HF Data Library** (https://hfdatalibrary.com, maintained by
Ahmed Elkassabgi, CC-BY-4.0, for academic research, not for trading). The Hugging
Face repo `elkassabgi/hfdatalibrary` holds documentation only.

1. Register for a free API key at https://hfdatalibrary.com (name, institution,
   email, role, country). Keys expire every 30 days.
2. **Never commit the key.** Keep it outside the repo, for example in
   `~/.config/hfdatalibrary/token` (`chmod 600`).
3. Download a ticker (full history, "clean" version, about 5 to 7 MB):
   ```bash
   mkdir -p data/scratch/tickers
   curl -H "X-API-Key: $(cat ~/.config/hfdatalibrary/token)" \
        -o data/scratch/tickers/AAPL_clean.parquet \
        "https://api.hfdatalibrary.com/v1/bars/AAPL?version=clean"
   ```
4. Available tickers (about 1,250, public): `https://api.hfdatalibrary.com/v1/symbols`.
   Limits: 100 downloads per minute.

Files have columns `datetime, Open, High, Low, Close, Volume, source`. `BarLoader`
converts them to a `datetime` index with lowercase `open, high, low, close,
volume`. `data/` and `.venv/` are git-ignored.

## Design

```
BarLoader ──bars──▶ Backtester ──▶ BacktestResult ──▶ Metrics
                      │  ▲  ▲  ▲        (equity, trades)    (summary stats)
                      │  │  │  └─ SmaCrossStrategy  (indicators, want_long)
                      │  │  └──── PositionSizer     (shares from risk/stop/cap)
                      │  └─────── CostModel         (commission, slippage)
```

| Class | File | Responsibility |
|---|---|---|
| `BarLoader` | `data.py` | Read parquet, normalise columns/index, de-duplicate, resample to any pandas bar size (empty periods dropped). |
| `CostModel` | `costs.py` | Commission per share with a per-order minimum, plus fixed per-share slippage on each side. Frozen dataclass. |
| `PositionSizer` | `sizing.py` | `min(equity*risk/stop_distance, equity*max_frac/price)` rounded down to whole shares. Returns 0 for invalid input. |
| `SmaCrossStrategy` | `strategy.py` | Computes `sma_fast`, `sma_slow`, `atr`, and the boolean `want_long` (optionally also requiring close > long SMA). Pure function of the bars. |
| `Backtester` | `engine.py` | Bar-by-bar loop: fills, stops, equity marking. Returns a `BacktestResult` holding the equity `Series` and a list of `Trade`s. |
| `Trade` | `engine.py` | One round trip (fill prices after slippage, total commission, exit reason, net `pnl`). |
| `Metrics` | `metrics.py` | Total return, CAGR, daily Sharpe, max drawdown, buy-and-hold, win rate, profit factor, commissions. |
| `run.py` | `run.py` | Command line wrapper (`python -m algotrader.run`). |

Each class has one job and takes its collaborators in its constructor, so any of
them can be swapped or tested alone (for example a new strategy only needs an
`indicators(bars)` method returning `want_long`, plus `atr_mult`).

### Execution rules (no look-ahead)

1. Signals are computed on a bar's **close** and filled at the **next bar's open**.
2. Entry only on a fresh crossover (`want_long` going False to True). After a stop-out
   the strategy waits for the next crossover instead of re-entering at once.
3. Stop = `atr_mult * ATR` below the entry fill (ATR from the signal bar), checked
   against each bar's **low**. Fill is the stop price, or the open if the bar gaps
   below it.
4. A position still open on the last bar is closed at that bar's close.
5. Equity is marked to the close of every bar. Sharpe uses **daily** equity returns
   (annualised with sqrt(252), risk-free rate 0).

See `STRATEGY.md` for the full specification.

## Tests

```bash
.venv/bin/python -m pytest -v
```

| File | What it checks |
|---|---|
| `tests/test_costs_sizing.py` | Commission minimum/per-share, slippage direction, sizing by risk vs cap, rounding, invalid input. |
| `tests/test_data.py` | Column/index normalisation, sorting, duplicate handling, resample OHLCV aggregation, gap dropping, parquet loading. |
| `tests/test_strategy.py` | SMA/ATR values, warm-up is flat, up/down/flat markets, trend filter, parameter validation. |
| `tests/test_engine.py` | Fill at next open, signal exit, **no look-ahead** (future data cannot change earlier equity), intrabar stop and gap stop, no re-entry after stop, end-of-data close, commission and slippage effects, accounting identity (final equity = capital + sum of trade pnl), size cap, cash never negative. |
| `tests/test_metrics.py` | Return, drawdown, Sharpe against a manual calculation, daily vs per-bar aggregation, CAGR, trade statistics, buy-and-hold. |

Tests use small hand-built price series (`tests/helpers.py::make_bars`) so every
expectation can be checked by hand.

## Results so far

Default strategy (SMA 20/50, 1% risk, 2x ATR stop, 10% max position, IBKR
commissions, no slippage), $10,000 start, 3 tickers, Mar 2022 to Oct 2026. These
tickers are those in the legacy results file, not a chosen sample.

| Bar size | ADM | AMC | AMGN | Trades per ticker |
|---|---|---|---|---|
| 1 min | -82.1% | -68.8% | -67.3% | 3,500 to 4,300 |
| 5 min | -12.1% | -28.9% | -16.4% | about 1,000 |
| 30 min | -3.0% | +7.0% | -3.7% | about 160 |
| 1 day | +1.8% | -9.8% | +2.3% | 13 |
| Buy and hold | -2.3% | -98.2% | +88.3% | |

Reading: on short bars the crossover whipsaws constantly and the $1 minimum
commission (on roughly $1,000 positions) costs about 0.2% per round trip, which
dominates. On 30 min and daily bars the result is indistinguishable from noise.
**There is no demonstrated edge here**; this is a working, trustworthy baseline to
test better ideas against, not a profitable strategy. Three tickers is far too
small a sample for any conclusion.

## Removed legacy code

The repo originally held an agent-written pipeline (`backtest/`, `test_voladj*.py`,
`results/`). It was deleted because it could not be trusted (it remains in git history,
commit `7fc720e` and earlier):

- The script `backtest/sma_backtest_voladj.py` run unchanged on the real data lost 99% to
  100% on every ticker (about 10,000 trades each).
- Its published `results/summary_with_commission_all.csv` listed 3 tickers (not the 50
  claimed), two of them with just 2 trades, and could not be reproduced from any code.
- Its generating pipeline (`/home/hermes/run_pipeline.sh`) and ticker list were never in the repo.

## Limitations

- Single instrument at a time, long only, whole shares, no leverage or borrowing costs.
- IEX volume is a small share of the consolidated tape; fills are assumed at the
  quoted bar prices with no market impact or partial fills.
- Clean-version data has some bars removed by the provider's outlier filter.
- Prices are as provided; corporate-action handling is the provider's. Check before
  trusting long histories.
- Results are pre-tax. Not investment advice.

## License

Educational and research use. Data licence: see HF Data Library terms.
