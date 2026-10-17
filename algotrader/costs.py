"""Transaction costs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-share commission with a per-order minimum and a cap, pass-through fees,
    plus slippage and spread.

    Defaults are Interactive Brokers Pro fixed pricing: $0.005/share, $1 minimum, at
    most 1% of the trade value, exchange and regulatory fees included.
    `ibkr_tiered()` gives the Tiered plan (lower commission, fees passed through).

    Attributes:
        commission_per_share: dollars charged per share traded.
        min_commission: minimum commission in dollars per order.
        max_commission_frac: commission cap as a fraction of the trade value (needs the
            price; 0 = no cap).
        fee_per_share: pass-through fees per share (exchange, clearing), not subject
            to the minimum or the cap.
        sale_fee_rate: fee on the value of sales (SEC fee), sells only.
        sale_fee_per_share, sale_fee_cap: per-share fee on sales with a per-order cap
            (FINRA trading activity fee).
        slippage_per_share: dollars added to buy prices and subtracted from
            sell prices, to model crossing the spread.
        spread_bps: half-spread in basis points of the price, paid on each side
            (buys fill this much above the quote, sells this much below).
    """
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_per_share: float = 0.0
    spread_bps: float = 0.0
    max_commission_frac: float = 0.01
    fee_per_share: float = 0.0
    sale_fee_rate: float = 0.0
    sale_fee_per_share: float = 0.0
    sale_fee_cap: float = 0.0

    def commission(self, shares: float, price: float | None = None, sell: bool = False) -> float:
        """Commission and fees in dollars for one order of `shares` at `price` (0 if no
        shares). Without a price the cap and the value-based sale fee are not applied."""
        if shares <= 0:
            return 0.0
        comm = max(shares * self.commission_per_share, self.min_commission)
        if price is not None and self.max_commission_frac > 0:
            comm = min(comm, self.max_commission_frac * shares * price)
        fees = shares * self.fee_per_share
        if sell:
            if price is not None:
                fees += self.sale_fee_rate * shares * price
            taf = shares * self.sale_fee_per_share
            fees += min(taf, self.sale_fee_cap) if self.sale_fee_cap > 0 else taf
        return comm + fees

    def buy_price(self, price: float) -> float:
        """Fill price for a buy at quoted `price`, after slippage."""
        return price * (1 + self.spread_bps / 1e4) + self.slippage_per_share

    def sell_price(self, price: float) -> float:
        """Fill price for a sell at quoted `price`, after slippage."""
        return price * (1 - self.spread_bps / 1e4) - self.slippage_per_share


def ibkr_tiered(spread_bps: float = 0.0, exchange_fee: float = 0.003,
                clearing_fee: float = 0.0002) -> CostModel:
    """IBKR Pro Tiered, lowest volume tier (<= 300k shares a month): $0.0035/share,
    $0.35 minimum, 1% cap, plus exchange (taking liquidity, ~$0.003/share) and clearing
    (~$0.0002/share) fees, SEC fee on sales (~$20.6 per $1M) and FINRA TAF on sales
    (~$0.000166/share, capped ~$8.30). Rates from the investment-portfolio session's
    summary of IBKR's pricing page (2026-10-16); they change, check before relying on them."""
    return CostModel(commission_per_share=0.0035, min_commission=0.35, spread_bps=spread_bps,
                     max_commission_frac=0.01, fee_per_share=exchange_fee + clearing_fee,
                     sale_fee_rate=0.0000206, sale_fee_per_share=0.000166, sale_fee_cap=8.30)
