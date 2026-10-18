"""Combined-signal and market-trend experiments for the long history (roadmap item 9).

Kept out of `longrun.py` so the other long results stay current; run with
`python -m algotrader.longrun_combo [names] --out data/long_combo` (same options as
`python -m algotrader.longrun`). The profitability ingredient needs the SEC facts
(`data/edgar`), as `value_long` and friends do.

Pre-registered (all counted by `trials`): combined signal long (top 20/50), combined long-short,
combined long without profitability (price ingredients only, no SEC data), and momentum or the
combined signal held only while SPY is above its 200-day mean.
"""
from . import longrun
from .combo import CombinedSignal, TrendMomentum
from .longrun import BENCHMARK, LongExperiment


def combo_panels(frames: dict, members=None) -> dict:
    """Profitability panel from the SEC facts plus SPY's closes."""
    key = ("combo", id(frames))
    if key not in longrun._PANELS:
        panels = dict(longrun.fundamental_panels(frames, members)["panels"])
        panels["market"] = frames[BENCHMARK]["close"]
        longrun._PANELS[key] = {"panels": {k: panels[k] for k in ("gross_profitability", "market")}}
    return longrun._PANELS[key]


def market_panels(frames: dict, members=None) -> dict:
    return {"panels": {"market": frames[BENCHMARK]["close"]}}


EXPERIMENTS = [
    LongExperiment("combo_long", "Long the best blend of momentum, low vol, 52-week high, profitability",
                   CombinedSignal, {"every": "M"}, {"top_n": [20, 50]}, prepare=combo_panels),
    LongExperiment("combo_ls_long", "Long best, short worst blend of the four signals",
                   CombinedSignal, {"every": "M"}, {"top_n": [20, 50], "short_n": [20, 50]},
                   allow_short=True, prepare=combo_panels),
    LongExperiment("combo_price_long", "Blend of the three price signals (no SEC data)",
                   CombinedSignal, {"every": "M", "components": ("momentum", "low_vol", "high52"),
                                    "min_components": 2}, {"top_n": [20, 50]}, prepare=market_panels),
    LongExperiment("combo_trend_long", "Combined signal held only while SPY is above its 200-day mean",
                   CombinedSignal, {"every": "M", "trend_days": 200}, {"top_n": [20, 50]},
                   prepare=combo_panels),
    LongExperiment("xs_momentum_trend_long", "Momentum held only while SPY is above its 200-day mean",
                   TrendMomentum, {"every": "M", "trend_days": 200},
                   {"lookback": [126, 252], "skip": [0, 21], "top_n": [20, 50]}, prepare=market_panels),
]


def main(argv=None):
    for e in EXPERIMENTS:
        if e.name not in longrun.BY_NAME:
            longrun.EXPERIMENTS.append(e)
            longrun.BY_NAME[e.name] = e
    return longrun.main(argv)


if __name__ == "__main__":
    main()
