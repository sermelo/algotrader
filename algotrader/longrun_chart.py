"""Price-chart CNN experiments for the long history (roadmap S9).

Kept out of `longrun.py` so adding them does not change the fingerprint of the other long
experiments; run with `python -m algotrader.longrun_chart` (same options and output as
`python -m algotrader.longrun`; it registers the experiments there and calls its `main`).

The CNN scores every member at each month end from a chart of its last 20 days
(`chartcnn.py`); one model is fitted every January on the previous 6 years, so the first
scores are for 2013 (trained on 2008-2012) and the first yearly fold of the parameter
choice (top N, bottom N) is 2014.
"""
import os
from pathlib import Path

from . import longrun
from .chartcnn import ChartCNN, ChartCNNModel
from .longrun import BENCHMARK, LongExperiment

FIRST_SCORE_YEAR = 2013
CHART = dict(d=20, h=20, channels=(16, 32, 64), train_years=6, stride=20, epochs=5, patience=2)


def chart_cnn_panels(frames: dict, members=None) -> dict:
    """Walk-forward month-end scores of the chart CNN (computed once per set of frames);
    slow: run on the VPS (`CHART_THREADS` sets the torch threads)."""
    key = ("chart", id(frames))
    if key in longrun._PANELS:
        return longrun._PANELS[key]
    import pandas as pd
    stocks = {t: f for t, f in frames.items() if t != BENCHMARK}
    model = ChartCNNModel(**CHART, threads=int(os.environ.get("CHART_THREADS", "0")) or None)
    scores = model.scores(stocks, members, FIRST_SCORE_YEAR)
    Path("data/long_panels").mkdir(parents=True, exist_ok=True)           # for diagnostics
    scores.to_parquet("data/long_panels/chart_cnn.parquet")
    pd.DataFrame(model.log, columns=["epoch", "validation_loss"]).to_csv(
        "data/long_panels/chart_cnn_training.csv", index=False)
    longrun._PANELS[key] = {"panels": {"chart_cnn": scores}}
    return longrun._PANELS[key]


EXPERIMENTS = [
    LongExperiment("chart_cnn_long", "Long the N highest chart-CNN probabilities (JKX 2023)",
                   ChartCNN, {"every": "M"}, {"top_n": [20, 50]},
                   prepare=chart_cnn_panels, first_fold=FIRST_SCORE_YEAR + 1),
    LongExperiment("chart_cnn_ls_long", "Long highest, short lowest chart-CNN probabilities",
                   ChartCNN, {"every": "M"}, {"top_n": [20, 50], "short_n": [20, 50]},
                   allow_short=True, prepare=chart_cnn_panels, first_fold=FIRST_SCORE_YEAR + 1),
]


def main(argv=None):
    for e in EXPERIMENTS:
        if e.name not in longrun.BY_NAME:
            longrun.EXPERIMENTS.append(e)
            longrun.BY_NAME[e.name] = e
    return longrun.main(argv)


if __name__ == "__main__":
    main()
