# algotrader

A small, tested backtester for long-only strategies (SMA crossover, decision tree)
on US equity minute bars. Written to replace an earlier, unverified set of scripts (see
[Removed legacy code](#removed-legacy-code)).

- `algotrader/` is the package, in a handful of small classes (see [Design](#design)).
- `tests/` has the unit tests (`pytest`, 71 tests, about 3 seconds).
- `STRATEGY.md` is the exact strategy and execution rules.
- `R&D/` has summaries and full Markdown text of research papers on intraday trading
  (`R&D/index.md`). Their results are the authors' claims, not reproduced here.

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `pyproject.toml`
declares them, `uv.lock` pins exact versions and `.python-version` pins Python 3.10.

```bash
uv sync              # create .venv with the locked runtime + dev dependencies
uv run pytest        # run the unit tests
```

`uv add <pkg>` (or `uv add --dev <pkg>`) adds a dependency and updates the lock file;
commit both. Without uv: `pip install -e . pytest`.

Get data (next section), then:

```bash
uv run python -m algotrader.run --data-dir data/scratch/tickers \
    --tickers ADM AMC AMGN --timeframe 5min --out data/results/summary_5min.csv

# decision tree: one model per ticker, fitted before --train-end, tested from it on
uv run python -m algotrader.run --data-dir data/scratch/tickers --strategy tree \
    --train-end 2024-01-01 --timeframe 1min --out data/results/tree_1min.csv
```

Omit `--tickers` to run every `*_clean.parquet` in the folder. `--help` lists all
options (`--strategy`, `--train-end`, `--max-depth`, `--fast`, `--slow`,
`--trend-filter`, `--atr-period`, `--atr-mult`, `--risk`, `--max-position`,
`--commission`, `--min-commission`, `--slippage`, `--capital`).

Or from Python:

```python
from algotrader import BarLoader, Backtester, Metrics

bars = BarLoader("data/scratch/tickers").load("AMGN", timeframe="30min")
result = Backtester().run(bars)            # default: SMA 20/50, 1% risk, IBKR commissions
print(Metrics(result, bars).summary())
print(result.trades_frame().head())

from algotrader import DecisionTreeStrategy
bars = BarLoader("data/scratch/tickers").load("AMGN", timeframe="1min")
train, test = bars[:"2023-12-31"], bars["2024-01-01":]
tree = DecisionTreeStrategy().fit(train)   # learn only from the earlier period
print(Metrics(Backtester(tree).run(test), test).summary())
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

Files have columns `datetime, Open, High, Low, Close, Volume, source`, with naive
US Eastern timestamps covering the regular session only (09:30 to 15:59).
`BarLoader` converts them to a `datetime` index with lowercase `open, high, low,
close, volume`. Intraday resampling is anchored to the 09:30 open, so `1h` bars run
09:30 to 10:30 and so on (the last bar of the day, 15:30 to 16:00, is a half bar).
`data/` and `.venv/` are git-ignored.

## Design

```
BarLoader ──bars──▶ Backtester ──▶ BacktestResult ──▶ Metrics
                      │  ▲  ▲  ▲        (equity, trades)    (summary stats)
                      │  │  │  └─ Strategy          (indicators, want_long)
                      │  │  └──── PositionSizer     (shares from risk/stop/cap)
                      │  └─────── CostModel         (commission, slippage)
```

| Class | File | Responsibility |
|---|---|---|
| `BarLoader` | `data.py` | Read parquet, normalise columns/index, de-duplicate, resample to any pandas bar size (intraday sizes anchored to 09:30, empty periods dropped). |
| `CostModel` | `costs.py` | Commission per share with a per-order minimum, plus fixed per-share slippage on each side. Frozen dataclass. |
| `PositionSizer` | `sizing.py` | `min(equity*risk/stop_distance, equity*max_frac/price)` rounded down to whole shares. Returns 0 for invalid input. |
| `Strategy` | `strategy.py` | Protocol the engine relies on: `indicators(bars)` returning `want_long` and `atr` columns, plus an `atr_mult` attribute. |
| `SmaCrossStrategy` | `strategy.py` | Computes `sma_fast`, `sma_slow`, `atr`, and the boolean `want_long` (optionally also requiring close > long SMA). Pure function of the bars. Validates its parameters. |
| `average_true_range`, `true_range` | `strategy.py` | ATR helper (simple mean of true range) and the true range itself, reusable by other strategies. |
| `DecisionTreeStrategy` | `tree.py` | Per-stock decision tree after arXiv:2405.13959: `fit(train_bars)` learns whether the next bar closes higher from nine features, `indicators(bars)` predicts `want_long`. Flat while any feature is undefined. |
| `tree_features` | `tree.py` | The nine causal features (returns, RSI, ADX, SMA and VWAP ratios, SMA/close correlation, volatilities). |
| `Backtester` | `engine.py` | Bar-by-bar loop: fills, stops, equity marking. Returns a `BacktestResult` holding the equity `Series`, the shares held at each bar (`position`) and a list of `Trade`s. |
| `Trade` | `engine.py` | One round trip (fill prices after slippage, total commission, exit reason, net `pnl`). |
| `Metrics` | `metrics.py` | Total return, CAGR, daily Sharpe, max drawdown, buy-and-hold, exposure (time in market), win rate, profit factor, commissions. |
| `run.py` | `run.py` | Command line wrapper (`python -m algotrader.run`). |

Each class has one job and takes its collaborators in its constructor, so any of
them can be swapped or tested alone. A new strategy only has to satisfy the
`Strategy` protocol: an `indicators(bars)` method returning `want_long` and `atr`,
plus an `atr_mult` attribute (`tests/test_engine.py::AlwaysLong` is a minimal example).

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
| `tests/test_data.py` | Column/index normalisation, sorting, duplicate handling, resample OHLCV aggregation, session-anchored intraday bars, calendar daily bars, gap dropping, parquet loading, missing-ticker error. |
| `tests/test_strategy.py` | SMA/ATR values, warm-up is flat, up/down/flat markets, trend filter, parameter validation. |
| `tests/test_engine.py` | Fill at next open, signal exit, **no look-ahead** (future data cannot change earlier equity), intrabar stop and gap stop, no re-entry after stop, end-of-data close, commission and slippage effects, accounting identity (final equity = capital + sum of trade pnl), size cap, cash never negative, position series matches trades, empty input rejected, custom strategy through the protocol. |
| `tests/test_metrics.py` | Return, drawdown, Sharpe against a manual calculation, daily vs per-bar aggregation, CAGR, trade statistics, profit factor without losses, exposure, buy-and-hold. |
| `tests/test_tree.py` | RSI and ADX on hand-checkable series, feature values, **features and predictions cannot see future bars**, label definition, learning a planted pattern and predicting it out of sample, flat warm-up, fit required, parameter validation, end to end through the engine. |
| `tests/test_run.py` | Command line end to end on a temporary parquet file (CSV output, cost options, tree mode with `--train-end`), and argument errors. |

Tests use small hand-built price series (`tests/helpers.py::make_bars`) so every
expectation can be checked by hand.

## Results so far

Default strategy (SMA 20/50, 1% risk, 2x ATR stop, 10% max position, IBKR
commissions, no slippage), $10,000 start, 3 tickers, Mar 2022 to Oct 2026. These
tickers are those in the legacy results file, not a chosen sample.

| Bar size | ADM | AMC | AMGN | Trades per ticker | Time in market |
|---|---|---|---|---|---|
| 1 min | -82.1% | -68.8% | -67.3% | 3,500 to 4,300 | 26% to 30% |
| 5 min | -12.1% | -28.9% | -16.4% | about 1,000 | 27% to 34% |
| 30 min | -3.0% | +7.0% | -3.7% | about 160 | 32% to 33% |
| 1 hour | -3.1% | -0.9% | +2.2% | 80 to 94 | 29% to 38% |
| 1 day | +1.8% | -9.8% | +2.3% | 13 | 13% to 47% |
| Buy and hold | -2.3% | -98.2% | +88.3% | | 100% |

Reading: on short bars the crossover whipsaws constantly and the $1 minimum
commission (on roughly $1,000 positions) costs about 0.2% per round trip, which
dominates. On 30 min and daily bars the result is indistinguishable from noise.
Buy and hold is not a like-for-like benchmark: positions are capped at 10% of
equity and are open only about a third of the time, so on average roughly 3% of
capital is in the market. That shrinks both the gains and the losses.
**There is no demonstrated edge here**; this is a working, trustworthy baseline to
test better ideas against, not a profitable strategy. Three tickers is far too
small a sample for any conclusion.

### Decision tree (arXiv:2405.13959)

Reproduction of the decision-tree paper (`R&D/`) with our engine: one tree per ticker
(depth 4, Gini), trained on Mar 2022 to Dec 2023, tested Jan 2024 to Oct 2026 (out of
sample). Same 3 tickers, $10,000 start. "Paper-like" means fully invested
(`--max-position 1 --risk 1`) with the stop effectively off (`--atr-mult 50`); the
paper used no costs and no stop. Buy and hold over the test period: ADM +14.7%,
AMC -50.4%, AMGN +51.8%.

1-minute bars (the paper's setting):

| Setup | ADM | AMC | AMGN | Trades (ADM / AMC / AMGN) |
|---|---|---|---|---|
| Paper-like, no costs | +27.6% | +10.5% | +92.4% | 7,052 / 337 / 43,076 |
| Paper-like, IBKR commissions | -99.3% | -67.9% | -97.4% | |
| Paper-like, IBKR + $0.01/share slippage | -99.5% | -97.5% | -97.5% | |
| Defaults (1% risk, 2 ATR stop, 10% cap), IBKR | -93.5% | -10.4% | -74.3% | |

Other bar sizes, paper-like:

| Bar size | No costs (ADM / AMC / AMGN) | IBKR (ADM / AMC / AMGN) | Time in market |
|---|---|---|---|
| 5 min | +47.6% / +68.9% / +4.9% | -58.0% / -100.0% / -10.3% | 4% to 9% |
| 30 min | -2.2% / -0.2% / +53.3% | -13.5% / -27.0% / +38.5% | 1% to 93% |
| 1 hour | -10.0% / +18.8% / +47.7% | -16.6% / -55.0% / +37.4% | 8% to 82% |

Reading: on 1-minute bars the tree does find something before costs (win rate about
52%, profit factor about 1.06), but it is worth only **0.15 to 0.35 basis points per
trade** (ADM, AMGN). The $1 minimum commission on a $10,000 position is 2 bp per round
trip, and a one-cent spread on a $60 stock is about 1.7 bp per side, so the edge is
roughly ten times too small to pay for trading. That holds at any account size, since
per-share commissions and spreads scale with the position. The gross edge may well be
bid-ask bounce in trade prints rather than anything tradeable (not tested). On 30 min
and 1 hour bars the tree is mostly long and tracks the market, never beating buy and
hold after costs. AMC's trees rarely go long (the training period was a steady
decline). **The paper's result does not survive realistic costs here.**

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
- The decision tree uses one fixed train/test split, not walk-forward retraining.
- IEX volume is a small share of the consolidated tape; fills are assumed at the
  quoted bar prices with no market impact or partial fills.
- Clean-version data has some bars removed by the provider's outlier filter.
- Prices are as provided; corporate-action handling is the provider's. Check before
  trusting long histories.
- Results are pre-tax. Not investment advice.

## License

Educational and research use. Data licence: see HF Data Library terms.
