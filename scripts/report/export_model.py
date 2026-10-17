"""Add the tables of the minute-model test (results/minute_model/*.md) to report_data.json as `model`.

    python scripts/report/export_model.py OUT
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
tables = {}
for name in ("summary", "placebo", "replay", "online", "audit"):
    heading, block = "", []
    lines = (Path("results/minute_model") / f"{name}.md").read_text().splitlines() + [""]
    for line in lines:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        if block:
            rows = [r for r in block if not all(set(c) <= set("-: ") for c in r)]
            tables[f"{name}: {heading}"] = {"header": rows[0], "rows": rows[1:]}
            block = []
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
summary = (Path("results/minute_model") / "summary.md").read_text()
data = json.loads((OUT / "report_data.json").read_text())
online = (Path("results/minute_model") / "online.md").read_text()
totals = re.findall(r"### Hold (\S+): threshold chosen.*?All (\d+) years together: (\d+) trades, ([+-][\d.]+) bp net per trade", online, re.S)
data["model"] = {"online_totals": [list(x) for x in totals], "tables": tables, "head": summary.split("\n")[2], "primary_pass": re.search(r"\*\*(\d+) of (\d+)\*\*", summary).groups(),
                 "replay_head": (Path("results/minute_model") / "replay.md").read_text().split("\n")[2]}
(OUT / "report_data.json").write_text(json.dumps(data, separators=(",", ":")))
print(len(tables), "tables:", *[k[:60] for k in tables], sep="\n  ")
