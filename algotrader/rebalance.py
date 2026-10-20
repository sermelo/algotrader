"""Weekly versus monthly rebalancing of the long-history strategies.

Question (sergio, 2026-10-20): would checking the same strategies every week instead of every month
earn more, once costs are paid? On the long point-in-time S&P 500 history, ten long-only strategies of
`concentration.strategies()` (seasonality is left out: its signal is the coming calendar month, so it
can only change at month ends) at top 5 and top 20, each rebalanced at month ends ('M') and at week ends
('W'), at gross, base and stress costs, plus the top 5 / keep-in-top-20 buffer at both frequencies.
Weights only change when the chosen names change, so a weekly check trades only when a name enters or
leaves. Short-term reversal, the classic weekly strategy, is already in `results/long/` (weekly).

Pre-registered rule: weekly is better for a strategy if weekly minus monthly (daily returns, base costs)
at top 5 or top 20 has a Fama-French alpha with t >= 2; better in general only if that holds for most
strategies. Test period 2012-01-01 on, as the other long results.

    python -m algotrader.rebalance run --out data/long_rebal        # backtests (VPS) -> returns.csv
    python -m algotrader.rebalance report --out results/long_rebal  # tables -> summary.md
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import concentration, longrun
from .concentration import TEST_START, BufferBand, _row
from .longrun import PointInTime
from .portfolio import PortfolioBacktester
from .stats import returns_stats

FREQS = ("M", "W")
SIZES = (5, 20)
BUFFER = (5, 20)
COST_VARIANTS = ("gross", "base", "stress")
SKIP = ("seasonality",)
DEPS = ("combo", "fundamentals", "pead", "seasonal", "longrun_combo")


def strategies() -> dict:
    return {n: e for n, e in concentration.strategies().items() if n not in SKIP}


def maker(exp, extra, every):
    """top_n -> strategy at rebalance frequency `every`."""
    fixed = dict(exp.fixed, every=every)
    return lambda k: exp.strategy(**fixed, **(extra or {}), top_n=k)


def jobs(make):
    """(label, strategy, cost variants) for one strategy at one frequency."""
    out = [(f"top{k}", make(k), COST_VARIANTS) for k in SIZES]
    out.append((f"buf{BUFFER[0]}-{BUFFER[1]}", BufferBand(make, *BUFFER), ("base", "stress")))
    return out


def n_configurations() -> int:
    return len(strategies()) * len(FREQS) * (len(SIZES) + 1)


def fingerprint() -> str:
    for mod in DEPS:
        importlib.import_module(f"algotrader.{mod}")
    h = hashlib.sha256(Path(__file__).read_bytes())
    for mod in ("concentration", "longrun", "portfolio", "costs", "stats", "factors") + DEPS:
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    return h.hexdigest()


def run(out: Path, history=longrun.HISTORY):
    fp = fingerprint()
    frames = longrun.load_frames(history)
    members = longrun.membership_matrix(longrun.parse_members((Path(history) / "sp500_members.csv").read_text()))
    out.mkdir(parents=True, exist_ok=True)
    cols, turnover = {}, {}
    for name, exp in strategies().items():
        extra = exp.prepare(frames, members) if exp.prepare else None
        for every in FREQS:
            for label, strat, costs in jobs(maker(exp, extra, every)):
                for var in costs:
                    res = PortfolioBacktester(PointInTime(strat, members), longrun.COSTS[var],
                                              longrun.CAPITAL).run(frames)
                    key = f"{name}|{label}|{every}"
                    cols[f"{key}|{var}"] = longrun._returns(res.equity, None, None)
                    if var == "base":
                        turnover[key] = longrun.turnover(res, TEST_START, None)
                print(f"{name} {label} {every}: Sharpe "
                      f"{returns_stats(cols[f'{name}|{label}|{every}|base'][TEST_START:])['sharpe']:.2f}", flush=True)
    df = pd.DataFrame(cols)
    df["SPY"] = frames[longrun.BENCHMARK]["close"].pct_change().reindex(df.index)
    df.to_csv(out / "returns.csv", float_format="%.6g")
    (out / "manifest.json").write_text(json.dumps(
        {"rebalance": {"code_sha256": fp, "data_sha256": longrun.data_fingerprint(history),
                       "configurations": n_configurations(), "turnover": turnover}}, indent=2, sort_keys=True))


# ---------------------------------------------------------------- report

def _sharpe(r):
    return returns_stats(r[r.index >= TEST_START].dropna())["sharpe"]


def _cagr(r):
    return returns_stats(r[r.index >= TEST_START].dropna())["cagr"]


def compare(df: pd.DataFrame, factors=None) -> pd.DataFrame:
    """Per strategy and label: monthly and weekly Sharpe (gross/base/stress), CAGR, and weekly minus
    monthly at base costs (mean per year, t, Fama-French alpha and t)."""
    from .french import factor_regression
    names = sorted({c.split("|")[0] for c in df.columns if "|" in c})
    labels = sorted({c.split("|")[1] for c in df.columns if "|" in c})
    rows = []
    for n in names:
        for lab in labels:
            if f"{n}|{lab}|W|base" not in df:
                continue
            row = {"strategy": n, "label": lab}
            for f in FREQS:
                for var in COST_VARIANTS:
                    c = f"{n}|{lab}|{f}|{var}"
                    row[f"sharpe_{f}_{var}"] = _sharpe(df[c]) if c in df else np.nan
                row[f"cagr_{f}"] = _cagr(df[f"{n}|{lab}|{f}|base"])
            d = (df[f"{n}|{lab}|W|base"] - df[f"{n}|{lab}|M|base"])
            d = d[d.index >= TEST_START].dropna()
            row["diff"] = d.mean() * 252
            row["diff_t"] = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan
            ff = factor_regression(d, factors) if factors is not None else {"alpha": np.nan, "alpha_t": np.nan}
            row["ff_alpha"], row["ff_t"] = ff["alpha"], ff["alpha_t"]
            rows.append(row)
    return pd.DataFrame(rows)


def report(out: Path, data_dir="data/french") -> str:
    from .french import load
    df = pd.read_csv(out / "returns.csv", index_col=0, parse_dates=True)
    man = json.loads((out / "manifest.json").read_text())["rebalance"]
    to = man["turnover"]
    factors = load(data_dir) if Path(data_dir).exists() else None
    cmp = compare(df, factors)
    names = sorted(cmp["strategy"].unique())
    tops = cmp[cmp["label"].str.startswith("top")]
    passes = sorted(n for n in names if (tops[tops["strategy"] == n]["ff_t"] >= 2).any())
    verdict = "weekly is better in general" if len(passes) > len(names) / 2 else "weekly is not better in general"
    end = df.index.max().date()
    years = (df.index.max() - TEST_START).days / 365.25

    avg = ["| Holding | Monthly Sharpe (gross / base / stress) | Weekly Sharpe (gross / base / stress) | "
           "Monthly CAGR | Weekly CAGR | Monthly turnover | Weekly turnover | Strategies where weekly is higher (base) |",
           "|---|---|---|---|---|---|---|---|"]
    for lab in [f"top{k}" for k in SIZES] + [f"buf{BUFFER[0]}-{BUFFER[1]}"]:
        s = cmp[cmp["label"] == lab]
        m = "/".join(f"{s[f'sharpe_M_{v}'].mean():.2f}" for v in COST_VARIANTS if s[f"sharpe_M_{v}"].notna().any())
        w = "/".join(f"{s[f'sharpe_W_{v}'].mean():.2f}" for v in COST_VARIANTS if s[f"sharpe_W_{v}"].notna().any())
        tm = np.mean([to[f"{n}|{lab}|M"] for n in s["strategy"]])
        tw = np.mean([to[f"{n}|{lab}|W"] for n in s["strategy"]])
        avg.append(f"| {lab} | {m} | {w} | {s['cagr_M'].mean():+.1%} | {s['cagr_W'].mean():+.1%} | {tm:.0%} | "
                   f"{tw:.0%} | {(s['sharpe_W_base'] > s['sharpe_M_base']).sum()} of {len(s)} |")

    per = ["| Strategy | Holding | Sharpe monthly -> weekly (gross) | (base) | (stress) | CAGR monthly -> weekly | "
           "Turnover monthly -> weekly | Weekly - monthly/yr (t) | FF alpha/yr (t) |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in cmp.iterrows():
        g = (f"{r['sharpe_M_gross']:.2f} -> {r['sharpe_W_gross']:.2f}" if pd.notna(r["sharpe_M_gross"]) else "")
        per.append(f"| `{r['strategy']}` | {r['label']} | {g} | {r['sharpe_M_base']:.2f} -> {r['sharpe_W_base']:.2f} | "
                   f"{r['sharpe_M_stress']:.2f} -> {r['sharpe_W_stress']:.2f} | {r['cagr_M']:+.1%} -> {r['cagr_W']:+.1%} | "
                   f"{to[f'{r.strategy}|{r.label}|M']:.0%} -> {to[f'{r.strategy}|{r.label}|W']:.0%} | "
                   f"{r['diff']:+.1%} ({r['diff_t']:.2f}) | {r['ff_alpha']:+.1%} ({r['ff_t']:.2f}) |")

    from .concentration import HEAD
    spy = df["SPY"]
    detail = ["### `op_profitability` (the best strategy of the concentration test)", "",
              HEAD.format(first="Version")]
    for f, fname in (("M", "monthly"), ("W", "weekly")):
        for lab in [f"top{k}" for k in SIZES] + [f"buf{BUFFER[0]}-{BUFFER[1]}"]:
            c = f"op_profitability|{lab}|{f}"
            if f"{c}|base" in df:
                detail.append(_row(f"{lab} {fname}", df[f"{c}|base"], spy, factors, df[f"{c}|stress"], to[c]))
    detail.append(_row("SPY (buy and hold)", spy, spy, factors))

    text = f"""# Weekly versus monthly rebalancing (long history, point-in-time S&P 500)

Generated by `python -m algotrader.rebalance report`; do not edit by hand. Ten long-only strategies of the
concentration test (seasonality left out: it can only change at month ends) at top 5, top 20 and top 5 kept
while in the top 20, rebalanced at month ends or at week ends (a weekly check trades only when a chosen name
changes), $1M, IBKR commission + 0 bp (gross) / 2 bp (base) / 5 bp (stress) half-spread, test period
{TEST_START.date()} to {end} ({years:.1f} years). Turnover one-sided per year. {man['configurations']}
configurations (counted in `results/trials.md`).

**Pre-registered rule:** weekly is better for a strategy if weekly minus monthly (base costs) at top 5 or top 20
has a Fama-French alpha with t >= 2; better in general only if that holds for most strategies.
**Result: {len(passes)} of {len(names)} strategies pass ({', '.join(passes) or 'none'}); {verdict}.**

## Average over the {len(names)} strategies

{chr(10).join(avg)}

## Each strategy

{chr(10).join(per)}

{chr(10).join(detail)}
"""
    (out / "summary.md").write_text(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("run", "report"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--history", default=str(longrun.HISTORY))
    p.add_argument("--french", default="data/french")
    a = p.parse_args(argv)
    if a.command == "run":
        run(a.out or Path("data/long_rebal"), a.history)
    else:
        print(report(a.out or Path("results/long_rebal"), a.french))


if __name__ == "__main__":
    main()
