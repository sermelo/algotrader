#!/bin/bash
# Roadmap D1 on the VPS: WIKI prices from Kaggle, then Tiingo within the free quota
# (stops at 500 symbols a month; rerun next month to continue), then the checks.
# Run from the repo root: nohup scripts/history_queue.sh > history.log 2>&1 &
export PATH=$HOME/.local/bin:$PATH
set -e
mkdir -p data/kaggle/wiki
if [ ! -f data/kaggle/wiki/quandl-wiki-prices-us-equites.zip ]; then
  (cd data/kaggle/wiki && uvx kaggle@latest datasets download marketneutral/quandl-wiki-prices-us-equites)
fi
[ -f data/history/sp500_members.csv ] || uv run python -m algotrader.history membership
uv run python -m algotrader.history wiki
uv run python -m algotrader.history tiingo
uv run python -m algotrader.history check
touch history_queue.done
