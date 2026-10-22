# Handoff (updated 2026-10-22)

State for the next Claude session. **What to do next is in `ROADMAP.md`** (research-based
priority list; keep it updated as items are done).

## Where things stand

- Everything is committed and pushed (`2c66786`, 2026-10-22); 336 tests pass; every committed
  result's fingerprint is current. Nothing is running on the VPS; `~/algotrader-work` there has
  the code of this commit.
- Last work: portfolio engine fix (ROADMAP 14, buys in column order, cash shortfall scaled
  evenly) and a full rerun of every portfolio experiment (`scripts/engine_rerun.sh`): no
  verdict changed. Before that: merger arbitrage (S15, fail), drawdown bounds (`drawdown.py`).
- Headline: 1,210 configurations tried; no strategy beats the luck bar or has significant
  factor alpha. The agreed recommendation stands: a UCITS global index fund.
- Open discussion (2026-10-22): sergio asked what to paper-trade. Proposed operating
  profitability, top 10, monthly (long history 2012-2026: Sharpe 0.98, +16.0%/yr vs SPY +15.0%,
  beta 0.83, FF alpha t 1.30), expecting roughly the market (plan on 6-8%/yr), as a test of
  execution and costs, not of edge. Not started: it needs a live signal script (today's
  target portfolio from current prices and the latest SEC facts; our data ends 2026-10-16),
  then `docs/ibkr-integration.md`. Alternative offered: combined signal with the SPY 200-day
  filter (+7.6%/yr, max drawdown -18%).
- Next dated task: on or after 2026-11-01 the Tiingo second batch (ROADMAP pending 2).

## VPS access

Hermes VPS (Oracle Cloud, 2 ARM cores, 12 GB RAM), user `hermes`, Tailscale IP
`100.88.160.76`, key `/home/sergio/Workspace/hermes_oci/ssh.key` (never print or copy
it). Work only in `~/algotrader-work`; do not touch `~/.hermes` or restart Hermes
services; never open ports. Ask the `Hermes` Claude session if anything changed.

```bash
K=/home/sergio/Workspace/hermes_oci/ssh.key; H=hermes@100.88.160.76
rsync -a --delete --exclude .venv --exclude data --exclude results --exclude .git \
  --exclude "*.log" --exclude "*.done" \
  -e "ssh -i $K" ./ $H:algotrader-work/
ssh -i $K $H 'cd ~/algotrader-work && uv sync && nohup uv run python -m algotrader.experiments <names> --workers 2 > run.log 2>&1 &'
rsync -a -e "ssh -i $K" $H:algotrader-work/results/ results/
```

Check fingerprints after copying results back:

```bash
uv run python -c "
import json; from algotrader import experiments as ex
for path in ['results/manifest.json', 'results/broad/manifest.json']:
    m = json.load(open(path))
    print(path, len(m), 'stale:', [k for k, v in m.items()
          if v['code_sha256'] != ex.experiment_fingerprint(ex.BY_NAME[k])])"
```

## Lessons

- Forking worker processes after scikit-learn's OpenMP threads started deadlocked a
  run: the pool uses `spawn`.
- Low memory locally: Claude Code stops background jobs; run heavy work on the VPS,
  daily-bar experiments are fine locally with `--workers 1`.
- Selecting on short windows let a setting with one trade win; hence the 5-trade
  activity rule and the walk-forward.
- New experiments: import their strategy module inside the registry section of
  `experiments.py` (as `factors.py` is) so no other experiment's fingerprint changes.
- Long runs: keep a watcher (`until ssh ... test -f <done marker>; do sleep 300; done`).
- sergio wants a Telegram message (`~/.claude/skills/notify-telegram/send.sh "..."`)
  when there is nothing left to do.
