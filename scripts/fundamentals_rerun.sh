#!/bin/bash
# Rerun every experiment whose fingerprint covers fundamentals.py, after the stale SEC values fix (TASKS 1-2), on the VPS:
# all long-history families (merger, core and broad do not depend on it). Two lanes on the two cores: deep arbitrage
# (slow, both price variants) and everything else, concentration first (it holds the paper-trading configuration).
# Outputs go to data/rerun_f/; merge deep arbitrage into long/ and long_nohf/ afterwards.
# Run from the repo root:  nohup scripts/fundamentals_rerun.sh > data/fundamentals_rerun.log 2>&1 &
export PATH=$HOME/.local/bin:$PATH
set -x
R=data/rerun_f
mkdir -p $R

LONG="xs_momentum_long xs_momentum_ls_long xs_momentum_buffer_long xs_reversal_long low_vol_long high52_long
  value_long value_ls_long profitability_long profitability_ls_long asset_growth_long asset_growth_ls_long
  op_profitability_long op_profitability_ls_long seasonality_long seasonality_ls_long pead_long pead_ls_long"

(   # lane A: deep arbitrage, separate output folders
    uv run python -m algotrader.longrun deep_arb_long --out $R/long_deep > $R/deep.log 2>&1
    uv run python -m algotrader.longrun deep_arb_long --no-hf --out $R/long_nohf_deep > $R/deep_nohf.log 2>&1
    touch $R/deep.done
) &

uv run python -m algotrader.concentration run --out $R/long_conc > $R/conc.log 2>&1
touch $R/conc.done
uv run python -m algotrader.longrun $LONG --out $R/long > $R/long.log 2>&1
uv run python -m algotrader.longrun $LONG --no-hf --out $R/long_nohf > $R/long_nohf.log 2>&1
uv run python -m algotrader.longrun_combo combo_long combo_ls_long combo_price_long combo_trend_long \
    xs_momentum_trend_long --out $R/long > $R/combo.log 2>&1
uv run python -m algotrader.longrun_stop --out $R/long_stop > $R/stop.log 2>&1
uv run python -m algotrader.rebalance run --out $R/long_rebal > $R/rebal.log 2>&1
uv run python -m algotrader.volsubset run --out $R/long_vol > $R/vol.log 2>&1
uv run python -m algotrader.midcap --out $R/long_mid > $R/mid.log 2>&1
uv run python -m algotrader.longrun_chart chart_cnn_long chart_cnn_ls_long --out $R/long > $R/chart.log 2>&1
touch $R/main.done
wait
touch $R/all.done
