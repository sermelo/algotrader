"""Add the minute-feature scans (results/minute_scan, results/minute_rules) to report_data.json as `scans`.

    python scripts/report/export_scans.py OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
cond = pd.read_csv("results/minute_scan/conditional.csv")
sig = pd.read_csv("results/minute_scan/signal_scan.csv")
rules = pd.read_csv("results/minute_rules/rule_check.csv")
r = lambda x, d=5: None if x is None or not np.isfinite(x) else round(float(x), d)
data = json.loads((OUT / "report_data.json").read_text())
keep = ["dvol20", "vol30", "range_bp", "rvol30", "abs_gap", "abs_ret_open", "price", "logdv20"]
c = cond[cond["condition"].isin(keep) & cond["label"].isin(["fwd30", "fwd90", "fwd240"])]
thr = [col for col in c.columns if col.startswith("up_")]
conditional = {}
for (name, lab), g in c.groupby(["condition", "label"]):
    g = g.sort_values("bin")
    conditional.setdefault(name, {})[lab] = {"value": [r(v, 4) for v in g["bin_value"]], "mean_abs": [r(v, 6) for v in g["mean_abs"]],
                                             "up": {t[3:]: [r(v, 4) for v in g[t]] for t in thr}}
tod = cond[cond["condition"] == "minute"]
tod = {lab: {"minute": [int(m) for m in g.sort_values("bin")["bin_value"]], "mean": [r(v, 6) for v in g.sort_values("bin")["mean"]],
             "mean_abs": [r(v, 6) for v in g.sort_values("bin")["mean_abs"]]} for lab, g in tod.groupby("label")}
top = sig.reindex(sig["ic_t"].abs().sort_values(ascending=False).index).head(12)
signals = [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in top.to_dict("records")]
n = len(rules)
null_max = float(np.sqrt(2 * np.log(n)))
best = rules.sort_values("net_t", ascending=False).head(8)
byfk = [g.sort_values("net_bp", ascending=False).iloc[0] for _, g in rules.groupby(["filter", "k"])]
cols = ["pick", "score", "filter", "k", "minute", "label", "gross_bp", "excess_bp", "excess_t", "net_bp", "net_t", "win_net", "net_years_positive", "years"]
fmt = lambda row: {k: (r(row[k], 3) if isinstance(row[k], (float, np.floating)) else (int(row[k]) if isinstance(row[k], (np.integer,)) else row[k])) for k in cols}
data["scans"] = {"conditional": conditional, "time_of_day": tod, "signals": signals, "n_signal_tests": len(sig),
                 "rules": {"n": n, "null_max_t": round(null_max, 2), "positive": int((rules["net_bp"] > 0).sum()),
                           "net_t_above_null": int((rules["net_t"] > null_max).sum()),
                           "excess_t_above_null": int((rules["excess_t"].abs() > null_max).sum()),
                           "best": [fmt(row) for _, row in best.iterrows()], "by_filter_k": [fmt(row) for row in byfk]}}
(OUT / "report_data.json").write_text(json.dumps(data, separators=(",", ":")))
print("scans:", len(conditional), "conditions,", len(signals), "signals,", n, "rule tests")
