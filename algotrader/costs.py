"""Transaction costs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-share commission with a per-order minimum, plus slippage and spread.

    Defaults mimic Interactive Brokers fixed pricing: $0.005/share, $1 minimum.

    Attributes:
        commission_per_share: dollars charged per share traded.
        min_commission: minimum dollars charged per order.
        slippage_per_share: dollars added to buy prices and subtracted from
            sell prices, to model crossing the spread.
        spread_bps: half-spread in basis points of the price, paid on each side
            (buys fill this much above the quote, sells this much below).
    """
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_per_share: float = 0.0
    spread_bps: float = 0.0

    def commission(self, shares: float) -> float:
        """Commission in dollars for one order of `shares` (0 if no shares)."""
        if shares <= 0:
            return 0.0
        return max(shares * self.commission_per_share, self.min_commission)

    def buy_price(self, price: float) -> float:
        """Fill price for a buy at quoted `price`, after slippage."""
        return price * (1 + self.spread_bps / 1e4) + self.slippage_per_share

    def sell_price(self, price: float) -> float:
        """Fill price for a sell at quoted `price`, after slippage."""
        return price * (1 - self.spread_bps / 1e4) - self.slippage_per_share
