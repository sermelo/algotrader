import numpy as np
import pandas as pd
import pytest

from algotrader import CostModel
from algotrader.factors import DualMomentum, High52Week, LowVolatility, SectorMomentum
from algotrader.portfolio import PortfolioBacktester, panel
from helpers import make_bars


def frames(**closes):
    return {t: make_bars(c, freq="1D") for t, c in closes.items()}


def last(w):
    return {k: v for k, v in w.iloc[-1].to_dict().items() if v}


def test_low_volatility_holds_the_calmest():
    calm = [100 + (0.1 if i % 2 else 0) for i in range(30)]
    wild = [100 + (5 if i % 2 else 0) for i in range(30)]
    f = frames(A=calm, B=wild, C=[100 + (2 if i % 2 else 0) for i in range(30)])
    w = LowVolatility(lookback=10, top_n=1, every="D").weights(panel(f, "close"))
    assert last(w) == {"A": 1.0}


def test_52_week_high_prefers_prices_near_their_high():
    f = frames(A=[10, 12, 14, 13.9], B=[10, 14, 12, 11], C=[10, 9, 8, 7])
    w = High52Week(lookback=3, top_n=1, every="D").weights(panel(f, "close"))
    assert last(w) == {"A": 1.0}


def test_sector_momentum_only_uses_sector_etfs():
    rising = list(range(100, 130))
    f = frames(XLK=rising, XLE=[100] * 30, SPY=[x * 2 for x in rising], XLF=[100] * 30)
    w = SectorMomentum(lookback=5, top_n=1, every="D").weights(panel(f, "close"))
    assert last(w) == {"XLK": 1.0}                    # SPY rose more but is not a sector


def test_dual_momentum_goes_to_safe_asset_when_everything_falls():
    up = list(range(100, 120))
    down = list(range(120, 100, -1))
    rising = frames(SPY=up, QQQ=[100] * 20, TLT=[100] * 20)
    falling = frames(SPY=down, QQQ=down, TLT=[100] * 20)
    strat = DualMomentum(lookback=5, top_n=1, every="D", risky=("SPY", "QQQ"))
    assert last(strat.weights(panel(rising, "close"))) == {"SPY": 1.0}
    assert last(strat.weights(panel(falling, "close"))) == {"TLT": 1.0}
    cash = DualMomentum(lookback=5, top_n=1, every="D", risky=("SPY", "QQQ"), safe=None)
    assert last(cash.weights(panel(falling, "close"))) == {}


def test_factor_strategies_run_through_the_portfolio_engine():
    rng = np.random.default_rng(0)
    f = {t: make_bars(list(100 + np.cumsum(rng.normal(0, 1, 300))), freq="1D")
         for t in ("XLF", "XLK", "XLE", "SPY", "QQQ", "TLT")}
    for strat in (LowVolatility(lookback=20, top_n=2), High52Week(lookback=60, top_n=2),
                  SectorMomentum(lookback=20, top_n=1), DualMomentum(lookback=20)):
        res = PortfolioBacktester(strat, CostModel(0, 0), 100_000).run(f)
        assert res.rebalances > 0 and res.equity.iloc[-1] > 0


@pytest.mark.parametrize("cls,kwargs", [(LowVolatility, {"lookback": 1}),
                                        (High52Week, {"top_n": 0}),
                                        (SectorMomentum, {"skip": 200, "lookback": 100}),
                                        (DualMomentum, {"every": "Q"})])
def test_validation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)
