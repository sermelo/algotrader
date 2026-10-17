import json, sys
import pandas as pd
from algotrader.data import BarLoader
from algotrader.universe import CORE, BROAD, ETFS, STOCKS, ORIGINAL, SECTOR_ETFS, SAMPLE
from algotrader.stats import expected_max_sharpe
L = BarLoader("data/scratch/tickers")
start, end = pd.Timestamp("2023-07-03"), pd.Timestamp("2026-10-16")
out = {}
for t in BROAD:
    b = L.load(t, "1D")
    p = b[(b.index >= start) & (b.index < end)]
    first = b.index[0]
    out[t] = {"ret": round(float(p["close"].iloc[-1] / p["open"].iloc[0] - 1), 4) if len(p) else None,
              "first": str(first.date()), "last": str(b.index[-1].date()),
              "ends_early": bool(b.index[-1] < pd.Timestamp("2026-10-01")),
              "group": ("index ETF" if t in ETFS else "large cap" if t in STOCKS else
                        "original 3" if t in ORIGINAL else "sector/bond/gold ETF" if t in SECTOR_ETFS else "random sample")}
years = 826 / 252
luck = {"years": round(years, 3), "n_uni": 389, "n_both": 778,
        "bar_uni": round(expected_max_sharpe(389, years), 3), "bar_both": round(expected_max_sharpe(778, years), 3),
        "se": round(1 / years ** 0.5, 3),
        "examples": {n: round(expected_max_sharpe(n, years), 2) for n in (1, 2, 10, 50, 100, 389, 778)}}
json.dump({"tickers": out, "core": CORE, "broad": BROAD, "luck": luck}, open(sys.argv[1] + "/tickers.json", "w"))
s = pd.Series({t: v["ret"] for t, v in out.items()}, dtype=float)
print("core mean", s[CORE].mean(), "broad mean", s.mean(), "SPY", s["SPY"], "QQQ", s["QQQ"])
print(s.sort_values().head(8).to_dict()); print(s.sort_values().tail(8).to_dict())
print([t for t, v in out.items() if v["ends_early"]]); print(luck)
