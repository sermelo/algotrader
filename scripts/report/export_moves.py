"""Add the price-move statistics (results/move_stats/move_stats.json) to report_data.json as `moves`.

    python scripts/report/export_moves.py OUT [move_stats.json]
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
src = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/move_stats/move_stats.json")
m = json.loads(src.read_text())
data = json.loads((OUT / "report_data.json").read_text())
r = lambda x, d=6: None if x is None else round(x, d)
data["moves"] = {
    "months": m["months"], "min_price": m["min_price"], "thresholds": m["thresholds"], "edges": m["edges"],
    "horizons": [int(h) for h in m["stats"]],
    "stats": {h: {k: (r(v) if isinstance(v, float) else v) for k, v in s.items() if k != "tails"} for h, s in m["stats"].items()},
    "hist": m["hist"],
    "by_year": {y: {h: r(s["tails"][1]["either"], 4) for h, s in hs.items()} for y, hs in m["by_year"].items()},
    "by_year_threshold": m["thresholds"][1],
}
(OUT / "report_data.json").write_text(json.dumps(data, separators=(",", ":")))
print("moves:", len(data["moves"]["horizons"]), "horizons,", len(data["moves"]["by_year"]), "years")
