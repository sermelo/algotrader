"""Position sizing."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSizer:
    """Risk-based sizing capped by a maximum fraction of equity.

    shares = min(equity * risk_per_trade / stop_distance,
                 equity * max_position_frac / price), rounded down to whole shares.

    Attributes:
        risk_per_trade: fraction of equity risked if the stop is hit (0.01 = 1%).
        max_position_frac: cap on position value as a fraction of equity.
    """
    risk_per_trade: float = 0.01
    max_position_frac: float = 0.10

    def shares(self, equity: float, price: float, stop_distance: float) -> int:
        """Whole shares to buy; 0 if inputs are invalid (non-positive or NaN)."""
        if not (equity > 0 and price > 0 and stop_distance > 0):
            return 0
        by_risk = equity * self.risk_per_trade / stop_distance
        by_cap = equity * self.max_position_frac / price
        return int(math.floor(min(by_risk, by_cap)))
