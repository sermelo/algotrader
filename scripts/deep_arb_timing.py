"""Time the deep-arbitrage steps on the WIKI history (run on the VPS: needs ~several GB).

    uv run python scripts/deep_arb_timing.py
"""
import time
from pathlib import Path

import pandas as pd

from algotrader import longrun
from algotrader.deep_arb import DeepArbModel, pca_residuals
from algotrader.history import parse_members

fr = longrun.load_frames()
fr.pop(longrun.BENCHMARK, None)
mem = longrun.membership_matrix(parse_members(Path("data/history/sp500_members.csv").read_text()))
closes = pd.DataFrame({t: f["close"] for t, f in fr.items()})
rets = closes.pct_change(fill_method=None)
m = mem.reindex(columns=closes.columns, fill_value=False).astype(float)
daily = m.reindex(index=closes.index.union(m.index)).ffill().reindex(closes.index).fillna(0.0) > 0.5
t = time.time()
eps, load = pca_residuals(rets, daily, k=5)
print("tickers", closes.shape[1], "residual days", len(load), "seconds", round(time.time() - t), flush=True)
sizes = [len(v[0]) for v in load.values()]
print("stocks per day", min(sizes), max(sizes), flush=True)
model = DeepArbModel(k=5, epochs=1)
model._col = {c: i for i, c in enumerate(rets.columns)}
pos = {d: i for i, d in enumerate(rets.index)}
days = [d for d in rets.index if d in load][30:1030]
E, Rn = eps.to_numpy(float), rets.to_numpy(float)
t = time.time()
model._batches(days, E, Rn, load, pos)
prep = time.time() - t
print("batch preparation, seconds", round(prep, 1), flush=True)
import torch
print("torch threads", torch.get_num_threads(), flush=True)
for epochs in (1, 3):
    model.epochs = epochs
    t = time.time()
    model._fit(days, E, Rn, load, pos)
    print(f"{epochs} epoch(s) incl. preparation, seconds", round(time.time() - t, 1), flush=True)
