import math
import pytest

from algotrader import CostModel, PositionSizer


class TestCostModel:
    def test_per_share_commission_above_minimum(self):
        assert CostModel().commission(1000) == 5.0

    def test_minimum_commission_applies(self):
        assert CostModel().commission(10) == 1.0

    def test_no_shares_no_commission(self):
        assert CostModel().commission(0) == 0.0

    def test_slippage_hurts_both_sides(self):
        m = CostModel(slippage_per_share=0.02)
        assert math.isclose(m.buy_price(100), 100.02)
        assert math.isclose(m.sell_price(100), 99.98)

    def test_spread_in_basis_points_on_each_side(self):
        m = CostModel(spread_bps=2.0, slippage_per_share=0.01)
        assert math.isclose(m.buy_price(100), 100.03)     # +2 bp of 100 = 0.02, + 0.01
        assert math.isclose(m.sell_price(50), 49.98)      # -2 bp of 50 = 0.01, - 0.01


class TestPositionSizer:
    def test_risk_bound(self):
        # risk $100 (1% of 10k) over a $5 stop -> 20 shares; cap would allow 100.
        assert PositionSizer(0.01, 0.10).shares(10_000, 10, 5) == 20

    def test_position_cap_bound(self):
        # risk would allow 1000 shares; cap is 10% of 10k / $50 = 20 shares.
        assert PositionSizer(0.01, 0.10).shares(10_000, 50, 0.1) == 20

    def test_rounds_down(self):
        assert PositionSizer(0.01, 1.0).shares(10_000, 10, 3) == 33

    def test_invalid_inputs_give_zero(self):
        s = PositionSizer()
        assert s.shares(10_000, 10, 0) == 0
        assert s.shares(10_000, 10, float("nan")) == 0
        assert s.shares(0, 10, 1) == 0
        assert s.shares(10_000, 0, 1) == 0


def test_commission_cap_and_sale_fees():
    from algotrader.costs import CostModel, ibkr_tiered
    fixed = CostModel()
    assert fixed.commission(100) == pytest.approx(1.0)                 # no price: no cap
    assert fixed.commission(1000, price=0.5) == pytest.approx(5.0)     # 1000*0.005, value 500
    assert fixed.commission(1000, price=0.05) == pytest.approx(0.5)    # capped at 1% of $50
    assert fixed.commission(100, price=50, sell=True) == fixed.commission(100, price=50)
    t = ibkr_tiered()
    buy = t.commission(66, price=150.0)
    sell = t.commission(66, price=150.0, sell=True)
    assert buy == pytest.approx(0.35 + 66 * 0.0032)                    # minimum + fees
    assert sell == pytest.approx(buy + 0.0000206 * 66 * 150 + 66 * 0.000166)
    big = t.commission(100_000, price=10.0, sell=True)
    assert big == pytest.approx(350 + 100_000 * 0.0032 + 0.0000206 * 1e6 + 8.30)  # TAF capped
    assert t.commission(0, price=10.0) == 0.0
