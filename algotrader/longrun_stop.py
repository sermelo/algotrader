"""Stop-loss test on the long S&P 500 history (`stoploss.py`).

Four long-only strategies at the parameters their walk-forward chose most often
(`results/long/*_folds.csv`): momentum 12-1 top 20, low volatility 252 days top 50, 52-week
high top 100, combined price signal top 50. Each gets a stop-loss grid: off, or 10, 15, 20,
25, 30% below the entry close (fixed) or below the highest close since entry (trailing),
11 configurations. The walk-forward chooses the stop every January from all earlier data
(off is one of the choices), as for every other parameter. For each fixed level it also
reports the whole test period (2012 on): Sharpe, CAGR, drawdown, turnover, how often the stop
fired, and the losses it avoided against the gains it gave up (each stop's return from the
stop to the next rebalance, times its weight, in % of the portfolio a year).

    python -m algotrader.longrun_stop --out data/long_stop          # -> copy to results/long_stop
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import longrun
from .combo import CombinedSignal
from .factors import High52Week, LowVolatility
from .longrun import LongExperiment, PointInTime
from .longrun_combo import market_panels
from .portfolio import CrossSectionalMomentum, PortfolioBacktester
from .stats import expected_max_sharpe, returns_stats
from .stoploss import stop_events, with_stop

STOPS = {"stop": [0.0, 0.10, 0.15, 0.20, 0.25, 0.30], "trail": [False, True]}
TEST_START = pd.Timestamp(f"{longrun.FIRST_FOLD}-01-01")

EXPERIMENTS = [
    LongExperiment("xs_momentum_stop_long", "Momentum 12-1, top 20, with a stop-loss",
                   with_stop(CrossSectionalMomentum),
                   {"every": "M", "lookback": 252, "skip": 21, "top_n": 20}, STOPS),
    LongExperiment("low_vol_stop_long", "Low volatility 252 days, top 50, with a stop-loss",
                   with_stop(LowVolatility), {"every": "M", "lookback": 252, "top_n": 50}, STOPS),
    LongExperiment("high52_stop_long", "52-week high, top 100, with a stop-loss",
                   with_stop(High52Week), {"every": "M", "lookback": 252, "top_n": 100}, STOPS),
    LongExperiment("combo_price_stop_long", "Combined price signal, top 50, with a stop-loss",
                   with_stop(CombinedSignal),
                   {"every": "M", "components": ("momentum", "low_vol", "high52"), "min_components": 2,
                    "top_n": 50}, STOPS, prepare=market_panels),
]
BY_NAME = {e.name: e for e in EXPERIMENTS}


def fingerprint(exp) -> str:
    h = hashlib.sha256(longrun.fingerprint(exp).encode())
    h.update(Path(__file__).read_bytes())
    for mod in ("stoploss", "combo", "factors", "longrun_combo"):
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    return h.hexdigest()


class _Closes:
    def weights(self, closes):
        return closes


def _max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def levels(exp: LongExperiment, frames: dict, members: pd.DataFrame) -> pd.DataFrame:
    """Each grid point over the whole test period: performance and the stops' accounting."""
    extra = exp.prepare(frames, members) if exp.prepare else None
    closes = pd.DataFrame({t: f["close"] for t, f in frames.items()}).sort_index()
    seen = PointInTime(_Closes(), members).weights(closes)       # members only, as the strategy sees
    base = exp.make({"stop": 0.0, "trail": False}, extra)
    w = base.weights(seen)
    years = (closes.index >= TEST_START).sum() / 252
    rows = []
    for p in exp.combos():
        res = PortfolioBacktester(PointInTime(exp.make(p, extra), members), longrun.COSTS["base"],
                                  longrun.CAPITAL).run(frames)
        r = longrun._returns(res.equity, TEST_START, None)
        st = returns_stats(r)
        ev = stop_events(w, seen, p["stop"], p["trail"], exp.fixed.get("every")) if p["stop"] else None
        if ev is not None:
            ev = ev[ev["date"] >= TEST_START]
        gain = loss = 0.0
        n = rec = 0
        if ev is not None and len(ev):
            contrib = ev["weight"] * ev["after"]
            gain = float(contrib[contrib > 0].sum()) / years      # given up by selling
            loss = float(-contrib[contrib < 0].sum()) / years     # avoided by selling
            n, rec = len(ev), float((ev["after"] > 0).mean())
        rows.append({"stop": p["stop"], "trail": p["trail"], "sharpe": st["sharpe"], "cagr": st["cagr"],
                     "max_dd": _max_dd(r), "turnover": longrun.turnover(res, TEST_START, None),
                     "stops_per_year": n / years, "recovered": rec,
                     "loss_avoided": loss, "gain_given_up": gain, "net": loss - gain})
    return pd.DataFrame(rows)


def write_summary(out: Path):
    man = json.loads((out / "manifest.json").read_text())
    n_conf = sum(v["configurations"] for v in man.values())
    parts = []
    years = 0.0
    for name in sorted(man):
        wf = pd.read_csv(out / f"{name}_wf_returns.csv", index_col=0, parse_dates=True)
        folds = pd.read_csv(out / f"{name}_folds.csv")
        lv = pd.read_csv(out / f"{name}_levels.csv")
        r = wf["base"].dropna()
        years = max(years, len(r) / 252)
        st = returns_stats(r)
        chosen = folds["params"].map(lambda s: json.loads(s)).map(
            lambda p: "off" if not p["stop"] else f"{p['stop']:.0%}{' trailing' if p['trail'] else ''}")
        lines = [f"## `{name}`", "",
                 f"Walk-forward (stop chosen each January): Sharpe {st['sharpe']:.2f}, CAGR {st['cagr']:+.1%}, "
                 f"max drawdown {_max_dd(r):.1%}, stress Sharpe {returns_stats(wf['stress'].dropna())['sharpe']:.2f}. "
                 f"Chosen: " + ", ".join(f"{c} x{k}" for c, k in chosen.value_counts().items()) + ".", "",
                 "| Stop | Sharpe | CAGR | Max DD | Turnover | Stops/yr | Recovered | Loss avoided/yr | Gain given up/yr | Net/yr |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for _, x in lv.iterrows():
            label = "off" if not x["stop"] else f"{x['stop']:.0%}{' trailing' if x['trail'] else ''}"
            lines.append(f"| {label} | {x['sharpe']:.2f} | {x['cagr']:+.1%} | {x['max_dd']:.1%} | {x['turnover']:.0%} | "
                         f"{x['stops_per_year']:.0f} | {x['recovered']:.0%} | {x['loss_avoided']:.2%} | "
                         f"{x['gain_given_up']:.2%} | {x['net']:+.2%} |")
        parts.append("\n".join(lines))
    text = f"""# Stop-loss test (long history, point-in-time S&P 500)

Generated by `python -m algotrader.longrun_stop`; do not edit by hand. Each strategy at its most-chosen
parameters; a holding is sold at the next open after a close 10-30% below its entry close (or below its
highest close since entry, "trailing") and stays out until the next monthly rebalance. Base costs (IBKR + 2 bp).
Tables: each stop level over the whole test period from {TEST_START.date()} (not walk-forward). "Recovered":
share of stops after which the stock was higher at the next rebalance than where it was sold. "Loss avoided" /
"gain given up": the stopped holdings' moves from the stop to the next rebalance, times their weight, in % of
the portfolio a year, before costs and with the money in cash at 0%; "Net" = avoided - given up.
{n_conf} configurations (the luck bar for that many over {years:.1f} years is about
{expected_max_sharpe(n_conf, years):.2f}, alone; see `results/trials.md` for the total).

""" + "\n\n".join(parts) + "\n"
    (out / "summary.md").write_text(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*")
    p.add_argument("--out", default="results/long_stop")
    p.add_argument("--history", default=str(longrun.HISTORY))
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    if a.list:
        for e in EXPERIMENTS:
            print(f"{e.name:24s} {len(e.combos()):3d} configs  {e.description}")
        return
    frames = longrun.load_frames(a.history)
    members = longrun.membership_matrix(longrun.parse_members((Path(a.history) / "sp500_members.csv").read_text()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in a.names or list(BY_NAME):
        exp = BY_NAME[n]
        res = longrun.run_long(exp, frames, members)
        longrun.write(res, out, a.history)
        man = json.loads((out / "manifest.json").read_text())
        man[n]["code_sha256"] = fingerprint(exp)
        (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
        levels(exp, frames, members).to_csv(out / f"{n}_levels.csv", index=False, float_format="%.6g")
        print(f"{n}: walk-forward Sharpe {returns_stats(res['wf']['base'])['sharpe']:.2f}", flush=True)
    write_summary(out)


if __name__ == "__main__":
    main()
