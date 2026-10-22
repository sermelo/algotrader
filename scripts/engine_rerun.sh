#!/bin/bash
# Rerun every experiment that uses the portfolio engine, after the order fix (ROADMAP 14), on the VPS.
# Two lanes on the two cores: deep arbitrage (slow, both price variants) and everything else.
# Long-history outputs go to data/rerun/; core and broad experiments write results/ as usual.
# Run from the repo root:  nohup scripts/engine_rerun.sh > data/engine_rerun.log 2>&1 &
export PATH=$HOME/.local/bin:$PATH
set -x
R=data/rerun
mkdir -p $R

PORTFOLIO="xs_momentum_daily xs_reversal_daily xs_momentum_ls_daily pairs_daily xs_gbm_daily xs_gbm_ls_daily
  low_vol_daily high52_daily sector_momentum_daily dual_momentum_daily xs_momentum_buffer_daily
  xs_momentum_buffer_ls_daily trend_multi_daily trend_multi_long_daily"
LONG="xs_momentum_long xs_momentum_ls_long xs_momentum_buffer_long xs_reversal_long low_vol_long high52_long
  value_long value_ls_long profitability_long profitability_ls_long asset_growth_long asset_growth_ls_long
  op_profitability_long op_profitability_ls_long seasonality_long seasonality_ls_long pead_long pead_ls_long"

(   # lane A: deep arbitrage, separate output folders (merged into long/ and long_nohf/ afterwards)
    uv run python -m algotrader.longrun deep_arb_long --out $R/long_deep > $R/deep.log 2>&1
    uv run python -m algotrader.longrun deep_arb_long --no-hf --out $R/long_nohf_deep > $R/deep_nohf.log 2>&1
    touch $R/deep.done
) &

uv run python -m algotrader.experiments $PORTFOLIO --workers 1 > $R/core.log 2>&1
uv run python -m algotrader.experiments $PORTFOLIO --universe broad --workers 1 > $R/broad.log 2>&1
uv run python -m algotrader.longrun $LONG --out $R/long > $R/long.log 2>&1
uv run python -m algotrader.longrun $LONG --no-hf --out $R/long_nohf > $R/long_nohf.log 2>&1
uv run python -m algotrader.longrun_combo combo_long combo_ls_long combo_price_long combo_trend_long \
    xs_momentum_trend_long --out $R/long > $R/combo.log 2>&1
uv run python -m algotrader.longrun_chart chart_cnn_long chart_cnn_ls_long --out $R/long > $R/chart.log 2>&1
uv run python -m algotrader.longrun_stop --out $R/long_stop > $R/stop.log 2>&1
uv run python -m algotrader.concentration run --out $R/long_conc > $R/conc.log 2>&1
uv run python -m algotrader.rebalance run --out $R/long_rebal > $R/rebal.log 2>&1
uv run python -m algotrader.midcap --out $R/long_mid > $R/mid.log 2>&1
uv run python -m algotrader.mergerarb run --out $R/merger > $R/merger.log 2>&1
touch $R/main.done
wait
touch $R/all.done
