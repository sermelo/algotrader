#!/bin/bash
# Rerun every experiment on both universes on the VPS (about 10-12 hours, 2 workers).
# Waits for any running longrun job first. Run from the repo root:
#   nohup scripts/full_rerun.sh > data/full_rerun.log 2>&1 &
export PATH=$HOME/.local/bin:$PATH
while pgrep -f "[a]lgotrader.longrun" > /dev/null; do sleep 300; done
uv run python -m algotrader.experiments --universe broad --workers 2 > data/run_broad.log 2>&1
uv run python -m algotrader.experiments --workers 2 > data/run_core.log 2>&1
uv run python -m algotrader.french report >> data/full_rerun.log 2>&1
uv run python -m algotrader.trials >> data/full_rerun.log 2>&1
touch data/full_rerun.done
