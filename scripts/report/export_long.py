"""Add the long-history (2012-2026, point-in-time S&P 500) results to report_data.json.

    python scripts/report/export_long.py OUT LONGRES
OUT has report_data.json (from export.py); LONGRES has long_hf/ and long_nohf/ (the
`python -m algotrader.longrun --out` folders: csv files and manifest.json).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from algotrader.longrun import BY_NAME
from algotrader.stats import alpha_beta, expected_max_sharpe, returns_stats, sharpe_interval

OUT, RES = Path(sys.argv[1]), Path(sys.argv[2])
FAMILY = {"Price and momentum": ("xs_momentum", "xs_reversal", "high52", "low_vol", "seasonality"),
          "Fundamentals": ("value", "profitability", "asset_growth", "op_profitability"),
          "Earnings news": ("pead",), "Deep learning": ("deep_arb", "chart_cnn")}
TICKERS = {"hf": 809, "nohf": 741}          # tickers loaded by each run (printed by longrun)


def r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def family(name):
    return next(f for f, ps in FAMILY.items() if any(name.startswith(p) for p in ps))


data = json.loads((OUT / "report_data.json").read_text())
long = {"experiments": {}, "variants": {"hf": "with Hugging Face prices", "nohf": "Tiingo and WIKI only"},
        "tickers": TICKERS}
configs = {}
for v, d in (("hf", RES / "long_hf"), ("nohf", RES / "long_nohf")):
    man = json.loads((d / "manifest.json").read_text())
    configs[v] = sum(m["configurations"] for m in man.values())
    for name in man:
        wf = pd.read_csv(d / f"{name}_wf_returns.csv", index_col=0, parse_dates=True)
        folds = pd.read_csv(d / f"{name}_folds.csv")
        b = wf["base"].dropna()
        st, (lo, hi) = returns_stats(b), sharpe_interval(b)
        ab = alpha_beta(b, wf["hold"]) if "hold" in wf else {}
        cum = (1 + b).cumprod().resample("W-FRI").last().dropna()
        e = long["experiments"].setdefault(name, {
            "description": BY_NAME[name].description, "family": family(name),
            "short": "_ls_" in name or name.endswith("_ls_long")})
        e[v] = {"sharpe": r(st["sharpe"], 3), "ci": [r(lo, 3), r(hi, 3)], "cagr": r(st["cagr"]),
                "ret": r(st["return"]), "mdd": r(st["max_drawdown"]),
                "gross": r(returns_stats(wf["gross"].dropna())["sharpe"], 3),
                "stress": r(returns_stats(wf["stress"].dropna())["sharpe"], 3),
                "alpha": r(ab.get("alpha")), "alpha_t": r(ab.get("alpha_t"), 2), "beta": r(ab.get("beta"), 3),
                "folds_pos": int((folds["sharpe"] > 0).sum()), "folds": int(len(folds)),
                "turnover": r(folds["turnover"].mean(), 3), "mean_bp": r(b.mean() * 1e4, 2),
                "up_days": r((b > 0).mean(), 3), "days": int(len(b)),
                "yearly": {str(y): r((1 + x).prod() - 1) for y, x in b.groupby(b.index.year)},
                "curve": [[str(i.date()), r(x, 4)] for i, x in cum.items()]}
        if "hold" in wf and "hold" not in long:
            h = wf["hold"].dropna()
            hc = (1 + h).cumprod().resample("W-FRI").last().dropna()
            hs = returns_stats(h)
            long["hold"] = {"sharpe": r(hs["sharpe"], 3), "cagr": r(hs["cagr"]), "ret": r(hs["return"]),
                            "mdd": r(hs["max_drawdown"]), "mean_bp": r(h.mean() * 1e4, 2),
                            "up_days": r((h > 0).mean(), 3), "days": int(len(h)),
                            "first": str(h.index.min().date()), "last": str(h.index.max().date()),
                            "yearly": {str(y): r((1 + x).prod() - 1) for y, x in h.groupby(h.index.year)},
                            "curve": [[str(i.date()), r(x, 4)] for i, x in hc.items()]}
years = long["hold"]["days"] / 252
long["luck"] = {v: {"configs": n, "bar": r(expected_max_sharpe(n, years), 2)} for v, n in configs.items()}
data["long"] = long
(OUT / "report_data.json").write_text(json.dumps(data, separators=(",", ":")))
print(len(long["experiments"]), "experiments;", long["hold"]["first"], "to", long["hold"]["last"],
      long["hold"]["days"], "days; luck", long["luck"])
