# Strategy scorecard report

Builds the interactive HTML report published at
https://claude.ai/artifact/7UUHsVHmANnLaDjStWZmQd (private) from the committed
`results/`. All scripts take an output/work directory `OUT` (not in the repo).

```bash
OUT=/tmp/scorecard; mkdir -p $OUT
uv run python scripts/report/tuning.py core $OUT     # every grid point out of sample (~15 min locally)
uv run python scripts/report/tuning.py broad $OUT    # ~1.5 h; better on the VPS
uv run python scripts/report/tickers.py $OUT         # per-ticker buy-and-hold returns, luck bar
uv run python scripts/report/export.py $OUT          # -> $OUT/report_data.json
uv run python scripts/report/build.py $OUT           # needs report_template.html in $OUT
```

`build.py` reads `report_template.html` from `$OUT`: copy it there first
(`cp scripts/report/report_template.html $OUT/`). The output is
`$OUT/strategy-scorecard.html`. Dates (826 days, 3 Jul 2023 – 15 Oct 2026), the luck
bar (389 configurations) and some text in the template are hard-coded for the
2026-10-17 results: update them after a rerun.
