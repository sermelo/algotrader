"""Collect committed results + tuning analysis into one JSON for the report page."""
import glob, json, os, sys
import numpy as np
import pandas as pd

SP = sys.argv[1]
GROUP = {
    "Daily, per ticker": ["sma_daily", "sma_daily_overlays", "rsi2_daily", "rsi2_daily_overlays",
                          "tsmom_daily", "turn_of_month_daily", "pre_holiday_daily"],
    "Portfolio (cross-sectional)": ["xs_momentum_daily", "xs_momentum_ls_daily", "xs_momentum_buffer_daily",
                                    "xs_momentum_buffer_ls_daily", "xs_reversal_daily", "low_vol_daily",
                                    "high52_daily", "sector_momentum_daily", "dual_momentum_daily",
                                    "trend_multi_daily", "trend_multi_long_daily", "pairs_daily"],
    "Machine learning": ["gbm_daily", "gbm_30min", "xs_gbm_daily", "xs_gbm_ls_daily",
                         "tree_30min", "tree_1min"],
    "Intraday": ["overnight_1min", "overnight_1min_100k", "overnight_1min_100k_filters",
                 "overnight_1min_100k_weekday", "orb_5min", "orb_5min_100k", "orb_5min_100k_filters",
                 "orb_5min_100k_target", "noise_area_1min", "intraday_momentum_5min"],
}
group_of = {n: g for g, ns in GROUP.items() for n in ns}


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


out = {"experiments": {}, "hold": {}, "tuning": {}}
for uni, d in (("core", "results"), ("broad", "results/broad")):
    man = json.load(open(f"{d}/manifest.json"))
    hold_curve = None
    for name, m in man.items():
        w = m["walk_forward"]
        f = f"{d}/{name}_wf_returns.csv"
        rets = pd.read_csv(f, index_col=0, parse_dates=True)
        b = rets["base"].dropna()
        traded = b.abs().sum() > 0
        e = out["experiments"].setdefault(name, {"description": m["description"],
                                                  "timeframe": m["timeframe"],
                                                  "group": group_of.get(name, "Other")})
        stats = None
        if traded:
            cum = (1 + b).cumprod()
            wk = cum.resample("W-FRI").last().dropna()
            yearly = {str(y): r((1 + x).prod() - 1) for y, x in b.groupby(b.index.year)}
            stats = {
                "sharpe": r(w["base"]["sharpe"], 3), "ci": [r(v, 3) for v in w["sharpe_90"]],
                "gross": r(w["gross"]["sharpe"], 3), "stress": r(w["stress"]["sharpe"], 3),
                "ret": r(w["base"]["return"]), "cagr": r(w["base"]["cagr"]),
                "mdd": r(w["base"]["max_drawdown"]),
                "alpha": r(w["vs_hold"]["alpha"]), "alpha_t": r(w["vs_hold"]["alpha_t"], 2),
                "beta": r(w["vs_hold"]["beta"], 3),
                "folds_pos": sum(1 for v in w["fold_sharpe"].values() if v is not None and v > 0),
                "mean_bp": r(b.mean() * 1e4, 2), "median_bp": r(b.median() * 1e4, 2),
                "sd_bp": r(b.std() * 1e4, 1), "up_days": r((b > 0).mean(), 3),
                "flat_days": r((b == 0).mean(), 3),
                "best": r(b.max(), 4), "worst": r(b.min(), 4),
                "mean_up_bp": r(b[b > 0].mean() * 1e4, 1), "mean_down_bp": r(b[b < 0].mean() * 1e4, 1),
                "yearly": yearly, "fold_params": m["fold_params"],
                "curve": [[str(i.date()), r(v, 4)] for i, v in wk.items()],
            }
        e[uni] = stats
        if hold_curve is None:
            h = rets["hold"].dropna()
            hc = (1 + h).cumprod().resample("W-FRI").last().dropna()
            hold_curve = [[str(i.date()), r(v, 4)] for i, v in hc.items()]
            out["hold"][uni] = {"curve": hold_curve, "sharpe": r(w["hold"]["sharpe"], 3),
                                "ret": r(w["hold"]["return"]), "mean_bp": r(h.mean() * 1e4, 2),
                                "median_bp": r(h.median() * 1e4, 2), "up_days": r((h > 0).mean(), 3),
                                "cagr": r(w["hold"]["cagr"]), "mdd": r(w["hold"]["max_drawdown"]),
                                "yearly": {str(y): r((1 + x).prod() - 1) for y, x in h.groupby(h.index.year)}}
    tf = f"{SP}/tuning_{uni}.csv"
    if os.path.exists(tf):
        t = pd.read_csv(tf)
        for _, row in t.iterrows():
            pts = pd.read_csv(f"{SP}/tuning_{uni}_{row.exp}.csv", index_col=0)["fixed"].dropna()
            out["tuning"].setdefault(row.exp, {})[uni] = {
                "wf": r(row.wf, 3), "median": r(row.fixed_median, 3), "min": r(row.fixed_min, 3),
                "max": r(row.fixed_max, 3), "beats": r(row.wf_beats_pct, 3),
                "rho": r(row.rho_mean, 3), "n": int(row.combos),
                "points": [r(v, 3) for v in pts.sort_values()],
                "best": row.best_fixed if isinstance(row.best_fixed, str) else None}
trials = open("results/trials.md").read()
out["luck"] = {"per_universe": 1.64, "both": 1.75, "configs": 389}
out["universe"] = json.load(open(f"{SP}/tickers.json"))
json.dump(out, open(f"{SP}/report_data.json", "w"), separators=(",", ":"))
print(os.path.getsize(f"{SP}/report_data.json") // 1024, "KB")
