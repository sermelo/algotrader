# Handoff (updated 2026-10-16)

State for the next Claude session. **What to do next is in `ROADMAP.md`** (research-based
priority list; keep it updated as items are done).

## Where things stand

- Everything is committed and pushed: 28 experiments on the core (27 tickers,
  `results/`) and broad (107 tickers, `results/broad/`) universes, all fingerprints
  current, 158 tests passing.
- **Running on the VPS:** `scripts/history_queue.sh` (Tiingo daily prices, first ~500
  symbols of the free monthly quota; log `~/algotrader-work/history.log`, done marker
  `history_queue.done`). On or after 2026-11-01 run it again for the remaining ~216
  symbols, then `python -m algotrader.history check`, run `python -m algotrader.longrun`
  on the VPS, `python -m algotrader.french report --results results/long`, and commit
  `results/long/` (see ROADMAP "Long-history runner"). Nothing else is running.
- Earlier note: `~/algotrader-work` there holds the code as of commit
  `63456db` (without `factors.py`); rsync the working tree again before the next run.
- Headline: no strategy clears the luck bar (317 configurations, Sharpe 1.61) or has
  significant alpha. Most consistent: long-short momentum (alpha t 1.49 core, 1.52
  broad). See README "Experiments".

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
