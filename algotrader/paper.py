"""Paper trading of a long-history strategy on an IBKR paper account (sergio, 2026-10-22).

Strategy: `config/paper.json` (operating profitability, top 10, monthly, $15,000). It is the
backtest's rule run forward: on the last NYSE session of each month, after the close, the target is
the top N members by the signal (`live.target`); if the chosen names differ from the names held,
every holding is traded to equal weight of the account's value with market-on-open orders (IBKR
`OPG`), filled at the next session's open, as the backtest's next-open fills; if the names are the
same nothing is traded (holdings drift, as in the backtest). The first run with an empty state
builds the portfolio at once.

`daily` is the one cron job (every weekday after the US close): it records the account value, the
positions and the day's fills (with IBKR's own commissions) under `data/paper/`, then decides and,
with `--submit`, places the orders. Safety: the account must be a paper account (id DU...) in
paper mode (and the reverse in live mode); refuses when the account is worth more than
`max_equity` (paper balance not reset), when orders are still open, when a STOP file exists, when
more than `max_orders` orders or a buy over `max_order_slots` x account / top_n would go out, or when a
target stock has no price; one decision per day. Messages go to `notify_cmd` (Telegram).

    python -m algotrader.paper daily            # dry run: records, prints the orders it would place
    python -m algotrader.paper daily --submit   # the cron job
    python -m algotrader.paper status           # account, positions, last decision
"""
import argparse
import csv
import json
import math
import re
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

CONFIG = Path("config/paper.json")


@dataclass(frozen=True)
class Config:
    mode: str = "paper"                      # 'paper' (account DU...) or 'live'
    host: str = "127.0.0.1"
    port: int = 4002                         # IB Gateway paper port
    client_id: int = 17
    signal: str = "operating_profitability"
    largest: bool = True
    top_n: int = 10
    capital: float = 15_000.0                # what the paper account should be reset to (USD)
    max_equity: float = 30_000.0             # refuse above this (paper balance not reset)
    cash_buffer: float = 0.01                # keep 1% in cash: fills at the open differ from the close
    min_order_value: float = 100.0           # skip rebalancing trims smaller than this (exits always go)
    max_orders: int = 30
    max_order_slots: float = 1.5             # no single buy above 1.5 x equity / top_n
    out: str = "data/paper"
    notify_cmd: list = field(default_factory=list)   # e.g. [".../notify-telegram/send.sh"]

    def __post_init__(self):
        if self.mode not in ("paper", "live"):
            raise ValueError("mode must be 'paper' or 'live'")
        if self.top_n < 1 or not 0 <= self.cash_buffer < 0.5 or self.capital <= 0:
            raise ValueError("require top_n >= 1, 0 <= cash_buffer < 0.5, capital > 0")
        if self.max_equity < self.capital or not self.max_order_slots >= 1 or self.max_orders < 1:
            raise ValueError("require max_equity >= capital, max_order_slots >= 1, max_orders >= 1")

    @classmethod
    def load(cls, path=CONFIG):
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- decisions (pure)

def is_month_end(day, calendar=None) -> bool:
    """True if `day` is the last NYSE session of its month."""
    import exchange_calendars as xc
    cal = calendar or xc.get_calendar("XNYS")
    day = pd.Timestamp(day).normalize()
    if not cal.is_session(day):
        return False
    return cal.next_session(day).month != day.month


def needs_rebalance(target: list, held: dict) -> bool:
    """The backtest trades only when the chosen names change."""
    return set(target) != {t for t, q in held.items() if q}


@dataclass(frozen=True)
class Order:
    ticker: str
    qty: int                 # + buy, - sell
    price: float             # reference price used for sizing (last close)
    reason: str


def plan(target: list, held: dict, prices: dict, equity: float, cash_buffer=0.01,
         min_order_value=100.0) -> list:
    """Orders to hold `target` at equal weight of `equity` (less the buffer), whole shares, sells first.
    Names not in the target are sold entirely; trims smaller than `min_order_value` are skipped."""
    missing = [t for t in target if not prices.get(t, 0) > 0]
    if missing:
        raise ValueError(f"no price for {missing}")
    if not target:
        return [Order(t, -q, prices.get(t, float("nan")), "exit") for t, q in held.items() if q]
    slot = equity * (1 - cash_buffer) / len(target)
    want = {t: math.floor(slot / prices[t]) for t in target}
    orders = []
    for t in sorted(set(held) | set(target), key=lambda t: (t in target, t)):
        have, w = int(held.get(t, 0)), want.get(t, 0)
        d = w - have
        if d == 0:
            continue
        if t not in target:
            orders.append(Order(t, d, prices.get(t, float("nan")), "exit"))
        elif have == 0:
            orders.append(Order(t, d, prices[t], "enter"))
        elif abs(d) * prices[t] >= min_order_value:
            orders.append(Order(t, d, prices[t], "rebalance"))
    return sorted(orders, key=lambda o: o.qty > 0)          # sells first


def check(orders: list, equity: float, cfg: Config) -> list:
    """Reasons to refuse the plan (empty if it may go out)."""
    problems = []
    if equity > cfg.max_equity:
        problems.append(f"account worth ${equity:,.0f} > max_equity ${cfg.max_equity:,.0f}: reset the paper balance")
    if len(orders) > cfg.max_orders:
        problems.append(f"{len(orders)} orders > max_orders {cfg.max_orders}")
    limit = cfg.max_order_slots * equity / cfg.top_n
    for o in orders:
        if o.qty > 0 and o.qty * o.price > limit:
            problems.append(f"{o.ticker}: buy ${o.qty * o.price:,.0f} > {cfg.max_order_slots:g} slots (${limit:,.0f})")
    return problems


# ---------------------------------------------------------------- broker

def ib_symbol(ticker: str) -> str:
    return ticker.replace(".", " ")          # BRK.B -> BRK B


def from_ib_symbol(symbol: str) -> str:
    return symbol.replace(" ", ".")


ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"


def parse_ecb(xml: str) -> dict:
    """ECB daily reference rates: currency -> units per euro."""
    rates = {c: float(r) for c, r in re.findall(r"currency=['\"](\w{3})['\"]\s+rate=['\"]([\d.]+)['\"]", xml)}
    if "USD" not in rates:
        raise ValueError("no USD rate in the ECB file")
    return rates


def ecb_usd_per_eur(opener=urllib.request.urlopen) -> float:
    with opener(ECB_URL, timeout=30) as r:
        return parse_ecb(r.read().decode())["USD"]


def usd_rate(vals: dict, ecb=ecb_usd_per_eur) -> tuple:
    """(US dollars per unit of the account's base currency, where the rate came from).

    From IBKR's account values: ('ExchangeRate', 'USD') or, as a paper account with dollar cash reports
    it, ('$LEDGER-ExchangeRate', 'USD') (base units per USD); a euro account with neither uses the ECB
    reference rate. Anything else is refused rather than assumed to be 1."""
    base = next((cur for (tag, cur) in vals if tag == "NetLiquidation" and cur not in ("", "BASE")), None)
    if base == "USD":
        return 1.0, "base"
    for tag in ("ExchangeRate", "$LEDGER-ExchangeRate"):
        rate = vals.get((tag, "USD"))
        if rate and float(rate) > 0:
            return 1.0 / float(rate), "ibkr"
    if base == "EUR":
        return ecb(), "ecb"
    raise RuntimeError(f"no USD exchange rate for base currency {base!r}")


def usd_per_base(vals: dict, ecb=ecb_usd_per_eur) -> float:
    """US dollars per unit of the account's base currency (see `usd_rate`)."""
    return usd_rate(vals, ecb)[0]


class IbkrBroker:
    """The few calls the runner needs, on IB Gateway / TWS through ib_async."""

    def __init__(self, cfg: Config, readonly=True):
        from ib_async import IB
        self.ib = IB()
        self.ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=30, readonly=readonly)
        accounts = self.ib.managedAccounts()
        if len(accounts) != 1:
            raise RuntimeError(f"expected one account, got {len(accounts)}")
        self.account = accounts[0]

    def close(self):
        self.ib.disconnect()

    def _values(self) -> dict:
        return {(v.tag, v.currency): v.value for v in self.ib.accountValues(self.account)}

    def _usd_per_base(self, vals) -> float:
        return usd_per_base(vals)

    def fx(self) -> dict:
        """Base currency, account value in it, and the dollar rate used (with its source)."""
        vals = self._values()
        (_, cur), base = next(((k, float(v)) for k, v in vals.items() if k[0] == "NetLiquidation" and k[1] not in ("", "BASE")))
        rate, source = usd_rate(vals)
        return {"base_currency": cur, "equity_base": base, "usd_per_base": rate, "rate_source": source}

    def equity_usd(self) -> float:
        vals = self._values()
        base = next(float(v) for (tag, cur), v in vals.items() if tag == "NetLiquidation" and cur != "")
        return base * self._usd_per_base(vals)

    def cash_usd(self) -> float:
        vals = self._values()
        base = next(float(v) for (tag, cur), v in vals.items() if tag == "TotalCashValue" and cur != "")
        return base * self._usd_per_base(vals)

    def positions(self) -> dict:
        return {from_ib_symbol(p.contract.symbol): int(p.position)
                for p in self.ib.positions(self.account) if p.contract.secType == "STK" and p.position}

    def _contracts(self, tickers):
        from ib_async import Stock
        cs = [Stock(ib_symbol(t), "SMART", "USD") for t in tickers]
        self.ib.qualifyContracts(*cs)
        return {t: c for t, c in zip(tickers, cs) if c.conId}

    def closes(self, tickers) -> dict:
        """Last close of each ticker (delayed data is enough: it is the previous close)."""
        self.ib.reqMarketDataType(4)                     # delayed-frozen if no subscription
        cs = self._contracts(list(tickers))
        out = {}
        for t, tk in zip(cs, self.ib.reqTickers(*cs.values())):
            p = tk.close if tk.close and tk.close > 0 else tk.marketPrice()
            if p and p > 0 and not math.isnan(p):
                out[t] = float(p)
        return out

    def open_orders(self) -> list:
        return [f"{t.order.action} {t.order.totalQuantity:g} {from_ib_symbol(t.contract.symbol)}"
                for t in self.ib.openTrades()]

    def place_moo(self, order: Order) -> int:
        from ib_async import MarketOrder
        c = self._contracts([order.ticker])[order.ticker]
        o = MarketOrder("BUY" if order.qty > 0 else "SELL", abs(order.qty), tif="OPG",
                        orderRef=f"algotrader:{order.reason}", account=self.account)
        return self.ib.placeOrder(c, o).order.orderId

    def fills_today(self) -> list:
        out = []
        for f in self.ib.reqExecutions():
            e, cr = f.execution, f.commissionReport
            out.append({"exec_id": e.execId, "time": str(e.time), "ticker": from_ib_symbol(f.contract.symbol),
                        "side": e.side, "shares": float(e.shares), "price": float(e.price),
                        "commission": float(cr.commission) if cr and cr.commission else float("nan"),
                        "order_ref": e.orderRef})
        return out


# ---------------------------------------------------------------- the daily job

def _append_csv(path: Path, rows: list, key=None):
    if not rows:
        return
    seen = set()
    if key and path.exists():
        seen = {r[key] for r in csv.DictReader(path.open())}
    rows = [r for r in rows if not key or r[key] not in seen]
    if not rows:
        return
    fields = list(rows[0])
    if path.exists():
        with path.open(newline="") as fh:
            old = csv.DictReader(fh)
            header, kept = list(old.fieldnames or []), list(old)
        if header != fields:                             # columns added: rewrite with all of them
            fields = header + [f for f in fields if f not in header]
            with path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(kept + rows)
            return
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


def daily(cfg: Config, broker, day, submit=False, target_fn=None, calendar=None) -> str:
    """Record, decide, and (with `submit`) place the orders. Returns the message sent."""
    from . import live
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp(day).normalize()
    if (out / "STOP").exists():
        return f"paper {day.date()}: STOP file present, nothing done"
    paper_account = broker.account.startswith("DU")
    if paper_account != (cfg.mode == "paper"):
        raise RuntimeError(f"account {broker.account} does not match mode {cfg.mode}: refusing")

    equity, cash, held, fx = broker.equity_usd(), broker.cash_usd(), broker.positions(), broker.fx()
    _append_csv(out / "fills.csv", broker.fills_today(), key="exec_id")
    _append_csv(out / "equity.csv", [{"day": str(day.date()), "equity_usd": round(equity, 2),
                                      "cash_usd": round(cash, 2), "positions": json.dumps(held, sort_keys=True),
                                      "base_currency": fx["base_currency"], "equity_base": round(fx["equity_base"], 2),
                                      "usd_per_base": round(fx["usd_per_base"], 6), "rate_source": fx["rate_source"]}])
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    lines = [f"paper {day.date()}: account ${equity:,.0f} (cash ${cash:,.0f}), {len(held)} positions"]

    first = "target" not in state
    if state.get("last_decision") == str(day.date()) or not (first or is_month_end(day, calendar)):
        return "\n".join(lines)
    target_fn = target_fn or (lambda d: live.target(d, cfg.top_n, cfg.signal, cfg.largest, out=out)["tickers"])
    target = target_fn(day)
    lines.append(f"target ({cfg.signal} top {cfg.top_n}): {', '.join(target)}")
    if not needs_rebalance(target, held):
        lines.append("same names as held: no trades")
        orders = []
    else:
        opened = broker.open_orders()
        if opened:
            raise RuntimeError(f"orders still open, refusing to add more: {opened}")
        prices = broker.closes(sorted(set(target) | set(held)))
        orders = plan(target, held, prices, equity, cfg.cash_buffer, cfg.min_order_value)
        problems = check(orders, equity, cfg)
        if problems:
            raise RuntimeError("plan refused: " + "; ".join(problems))
        for o in orders:
            lines.append(f"  {'BUY ' if o.qty > 0 else 'SELL'} {abs(o.qty):5d} {o.ticker:6s} ~${o.price:,.2f} ({o.reason})")
    if submit:
        ids = [broker.place_moo(o) for o in orders]
        _append_csv(out / "orders.csv", [dict(asdict(o), day=str(day.date()), order_id=i) for o, i in zip(orders, ids)])
        state.update(target=target, last_decision=str(day.date()))
        state_path.write_text(json.dumps(state, indent=2))
        lines.append(f"{len(orders)} market-on-open orders placed" if orders else "nothing to place")
    else:
        lines.append("dry run: nothing placed (use --submit)")
    return "\n".join(lines)


def notify(cfg: Config, text: str):
    print(text, flush=True)
    if cfg.notify_cmd:
        try:
            subprocess.run([*cfg.notify_cmd, text], timeout=60, check=False)
        except OSError as e:
            print(f"notify failed: {e}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("daily", "status"))
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--day", default=str(pd.Timestamp.now(tz="America/New_York").date()))
    p.add_argument("--submit", action="store_true", help="place the orders (otherwise a dry run)")
    a = p.parse_args(argv)
    cfg = Config.load(a.config)
    broker = None
    try:
        broker = IbkrBroker(cfg, readonly=not a.submit)
        if a.command == "status":
            state = Path(cfg.out) / "state.json"
            fx = broker.fx()
            print(f"account {broker.account}: ${broker.equity_usd():,.0f}, cash ${broker.cash_usd():,.0f} "
                  f"({fx['base_currency']} {fx['equity_base']:,.0f} at {fx['usd_per_base']:.4f} USD, {fx['rate_source']})")
            print("positions:", broker.positions())
            print("open orders:", broker.open_orders())
            print("state:", state.read_text() if state.exists() else "none")
            return
        notify(cfg, daily(cfg, broker, a.day, submit=a.submit))
    except Exception as e:                                # every failure reaches the phone
        notify(cfg, f"paper {a.day}: ERROR {type(e).__name__}: {e}")
        raise
    finally:
        if broker is not None:
            broker.close()


if __name__ == "__main__":
    main()
