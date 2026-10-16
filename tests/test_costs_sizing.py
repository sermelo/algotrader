import math

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
