"""Out-of-sample value of tuning: every grid point's Sharpe in each walk-forward fold,
for the rule-based daily experiments, compared with the in-sample score the runner
used to choose. Usage: tuning.py core|broad"""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from algotrader import experiments as ex
from algotrader.data import BarLoader
from algotrader.portfolio import PortfolioBacktester
from algotrader.stats import portfolio_returns, returns_stats
from algotrader.universe import UNIVERSES

uni = sys.argv[1]
out_dir = "results" if uni == "core" else f"results/{uni}"
tickers = UNIVERSES[uni]
data = "data/scratch/tickers"
loader = BarLoader(data)
daily = {}
for t in tickers:
    try:
        daily[t] = loader.load(t, "1D")
    except FileNotFoundError:
        pass
market = daily.get(ex.MARKET)
folds = ex.fold_windows()


def sharpe(r):
    return returns_stats(r)["sharpe"] if len(r) else np.nan


rows = []
for exp in ex.EXPERIMENTS:
    if exp.timeframe != "1D" or exp.fit or len(exp.combos()) < 2:
        continue
    grid = pd.read_csv(f"{out_dir}/{exp.name}_grid.csv")
    port = isinstance(exp, ex.PortfolioExperiment)
    oos = {}      # (fold, params json) -> (sharpe, daily returns)
    for p in exp.combos():
        key = json.dumps(p, sort_keys=True)
        if port:
            res = PortfolioBacktester(exp.make(p), exp.costs_for("base"), exp.capital,
                                      exp.allow_short, exp.max_gross, exp.borrow_bps).run(daily)
            for s, e in folds:
                rec = ex._portfolio_summary(res, s, e)
                r = rec[1].pct_change().dropna() if rec else pd.Series(dtype=float)
                oos[(s.date().isoformat(), key)] = r
        else:
            curves = {s: {} for s, _ in folds}
            for t, bars in daily.items():
                full = ex._engine(exp, exp.make(p, market), "base").run(bars)
                for s, e in folds:
                    rec = ex._window_record(full, bars, s, e)
                    if rec:
                        curves[s][t] = rec[1]
            for s, _ in folds:
                oos[(s.date().isoformat(), key)] = (portfolio_returns(curves[s]) if curves[s]
                                                    else pd.Series(dtype=float))
    # per fold: rank correlation in-sample score vs OOS Sharpe; chosen vs median vs best
    rho, chosen_r, med_s, best_s, chosen_s = [], [], [], [], []
    for s, _ in folds:
        f = s.date().isoformat()
        g = grid[grid.fold == f].copy()
        g["oos"] = [sharpe(oos[(f, k)]) for k in g.params]
        k_ch, _ = ex._choose(g.to_dict("records"))
        pick = g.iloc[k_ch]
        if g.oos.notna().sum() > 2 and g.portfolio_sharpe.notna().sum() > 2:
            rho.append(spearmanr(g.portfolio_sharpe, g.oos, nan_policy="omit")[0])
        chosen_s.append(pick.oos); med_s.append(g.oos.median()); best_s.append(g.oos.max())
        chosen_r.append(oos[(f, pick.params)])
    # whole-period: each grid point held fixed through all folds
    fixed = {k: sharpe(pd.concat([oos[(s.date().isoformat(), k)] for s, _ in folds]))
             for k in grid.params.unique()}
    fixed_s = pd.Series(fixed)
    wf = sharpe(pd.concat(chosen_r))
    rank = (fixed_s < wf).mean()
    best_fixed = fixed_s.idxmax()
    pd.DataFrame({"fixed": fixed_s}).assign(exp=exp.name, wf=wf).to_csv(
        f"{sys.argv[2]}/tuning_{uni}_{exp.name}.csv")
    rows.append(dict(exp=exp.name, combos=len(exp.combos()), wf=wf,
                     fixed_median=fixed_s.median(), fixed_min=fixed_s.min(), fixed_max=fixed_s.max(),
                     wf_beats_pct=rank, rho_mean=np.nanmean(rho) if rho else np.nan,
                     best_fixed=best_fixed))
    print(f"{exp.name:28s} n={len(exp.combos()):3d} WF {wf:5.2f} | fixed grid points: "
          f"median {fixed_s.median():5.2f} [{fixed_s.min():5.2f}, {fixed_s.max():5.2f}] "
          f"WF beats {rank:4.0%} | IS-OOS rank corr {np.nanmean(rho):5.2f} | best fixed {best_fixed}",
          flush=True)


pd.DataFrame(rows).to_csv(f"{sys.argv[2]}/tuning_{uni}.csv", index=False)
