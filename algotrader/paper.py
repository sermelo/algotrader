"""Paper trading of long-history strategies on one IBKR paper account (sergio, 2026-10-22).

Strategies: the `books` of `config/paper.json` (operating profitability top 10 monthly and momentum
top 5 weekly, $15,000 each). Each book is the backtest's rule run forward on its own shares and cash:
on the last NYSE session of its period (month or week), after the close, the target is the top N
members by its signal (`live.target`, `live.momentum_target`); if the chosen names differ from the
names the book holds, every holding of the book is traded to equal weight of the book's value with
market-on-open orders (IBKR `OPG`), filled at the next session's open, as the backtest's next-open
fills; if the names are the same nothing is traded (holdings drift, as in the backtest). A book with
no state builds its portfolio at once.

The books' orders are netted into one order per stock (the account never sends a buy and a sell of
the same stock), and each fill is split back between the books (`allocate`): the books on the other
side of the net order trade with the others at the fill price. The books' shares must add up to the
account's positions every day, or nothing is traded. Cash that no book owns (dividends, interest,
currency differences) is reported as unallocated.

`daily` is the one cron job (every weekday after the US close): it records the account value, the
positions, the day's fills (with IBKR's own commissions) and each book's value under `data/paper/`,
then decides and, with `--submit`, places the orders. Safety: the account must be a paper account
(id DU...) in paper mode (and the reverse in live mode); refuses when the account is worth more than
`max_equity` (paper balance not reset) or less than the books' capital at their start, when orders
are still open, when the books do not add up to the account, when a STOP file exists, when more than
`max_orders` orders or a buy over `max_order_slots` x book value / top_n would go out, or when a
target stock has no price; one decision per book and day. Messages go to `notify_cmd` (Telegram).

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
MOMENTUM = "momentum"                        # the price signal (12-1 month return); others are SEC signals


@dataclass(frozen=True)
class Book:
    """One strategy's part of the account: its own shares and cash, decided on its own schedule."""
    name: str
    signal: str = "operating_profitability"  # a `fundamentals.signals` panel, or MOMENTUM
    top_n: int = 10
    every: str = "M"                         # decide on the last session of each month ('M') or week ('W')
    capital: float = 15_000.0                # the book's money at its start (USD)
    largest: bool = True

    def __post_init__(self):
        if not self.name or self.top_n < 1 or self.capital <= 0 or self.every not in ("M", "W"):
            raise ValueError("require a name, top_n >= 1, capital > 0, every 'M' or 'W'")


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
    books: tuple = ()                        # Book list; empty: one monthly book from the fields above

    def __post_init__(self):
        if self.mode not in ("paper", "live"):
            raise ValueError("mode must be 'paper' or 'live'")
        if self.top_n < 1 or not 0 <= self.cash_buffer < 0.5 or self.capital <= 0:
            raise ValueError("require top_n >= 1, 0 <= cash_buffer < 0.5, capital > 0")
        if self.max_equity < self.capital or not self.max_order_slots >= 1 or self.max_orders < 1:
            raise ValueError("require max_equity >= capital, max_order_slots >= 1, max_orders >= 1")
        names = [b.name for b in self.books]
        if len(set(names)) != len(names) or sum(b.capital for b in self.books) > self.max_equity:
            raise ValueError("book names must differ and their capital must not exceed max_equity")

    @classmethod
    def load(cls, path=CONFIG):
        raw = json.loads(Path(path).read_text())
        raw["books"] = tuple(Book(**b) for b in raw.get("books", ()))
        return cls(**raw)

    def book_list(self) -> tuple:
        return self.books or (Book("main", self.signal, self.top_n, "M", self.capital, self.largest),)


# ---------------------------------------------------------------- decisions (pure)

def is_period_end(day, every="M", calendar=None) -> bool:
    """True if `day` is the last NYSE session of its month ('M') or week ('W', Monday to Sunday, as the
    backtest's rebalance days)."""
    import exchange_calendars as xc
    cal = calendar or xc.get_calendar("XNYS")
    day = pd.Timestamp(day).normalize()
    if not cal.is_session(day):
        return False
    nxt = cal.next_session(day)
    return nxt.to_period(every) != day.to_period(every)


def is_month_end(day, calendar=None) -> bool:
    """True if `day` is the last NYSE session of its month."""
    return is_period_end(day, "M", calendar)


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


def check(orders: list, equity: float, cfg: Config, top_n=None) -> list:
    """Reasons to refuse the plan (empty if it may go out); `equity` and `top_n` of the book."""
    problems = []
    if equity > cfg.max_equity:
        problems.append(f"account worth ${equity:,.0f} > max_equity ${cfg.max_equity:,.0f}: reset the paper balance")
    if len(orders) > cfg.max_orders:
        problems.append(f"{len(orders)} orders > max_orders {cfg.max_orders}")
    limit = cfg.max_order_slots * equity / (top_n or cfg.top_n)
    for o in orders:
        if o.qty > 0 and o.qty * o.price > limit:
            problems.append(f"{o.ticker}: buy ${o.qty * o.price:,.0f} > {cfg.max_order_slots:g} slots (${limit:,.0f})")
    return problems


def net_orders(book_orders: dict) -> tuple:
    """{book: [Order]} -> (one Order per ticker with the summed quantity, sells first; {ticker: {book: qty}}).
    A ticker whose quantities cancel out gets no order."""
    alloc = {}
    for b, orders in book_orders.items():
        for o in orders:
            alloc.setdefault(o.ticker, {})[b] = alloc.get(o.ticker, {}).get(b, 0) + o.qty
    price = {o.ticker: o.price for orders in book_orders.values() for o in orders}
    reason = {}
    for b, orders in book_orders.items():
        for o in orders:
            reason.setdefault(o.ticker, []).append(f"{b}:{o.reason}")
    net = [Order(t, sum(a.values()), price[t], "+".join(reason[t])) for t, a in sorted(alloc.items())
           if sum(a.values())]
    return sorted(net, key=lambda o: o.qty > 0), alloc


def allocate(alloc: dict, filled: int, price: float, commission: float = 0.0) -> dict:
    """Split a net order's fill between the books: {book: (shares, cash change)}.

    `alloc` is each book's wanted quantity (signed), `filled` the shares the market filled (signed, the
    net order's side). Books on the other side of the net order (or all books, if it nets to zero)
    trade their whole quantity with the others at `price`; the books on its side share the market fill
    plus those crossed shares in proportion to their quantities (whole shares, the remainder to the
    largest) and pay the commission in the same proportion. Shares and cash add up to the fill."""
    net = sum(alloc.values())
    sign = (net > 0) - (net < 0)
    other = {b: q for b, q in alloc.items() if sign == 0 or q * sign <= 0}
    side = {b: abs(q) for b, q in alloc.items() if b not in other}
    avail = abs(filled) + sum(abs(q) for q in other.values())
    want = sum(side.values())
    got = {b: (q * avail) // want for b, q in side.items()} if want else {}
    if got:
        got[max(side, key=lambda b: (side[b], b))] += avail - sum(got.values())
    out = {b: (q, -q * price) for b, q in other.items()}
    for b, g in got.items():
        out[b] = (sign * g, -sign * g * price - (commission * g / avail if avail else 0.0))
    return out


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

    def history(self, tickers, duration="2 Y") -> pd.DataFrame:
        """Daily closes adjusted for splits and dividends (IBKR `ADJUSTED_LAST`) up to the last session,
        dates x tickers (a ticker without data is left out)."""
        cols = {}
        for t, c in self._contracts(list(tickers)).items():
            bars = self.ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting="1 day",
                                             whatToShow="ADJUSTED_LAST", useRTH=True, formatDate=1, timeout=120)
            if bars:
                cols[t] = pd.Series({pd.Timestamp(b.date): float(b.close) for b in bars})
        return pd.DataFrame(cols).sort_index()

    def open_orders(self) -> list:
        return [f"{t.order.action} {t.order.totalQuantity:g} {from_ib_symbol(t.contract.symbol)}"
                for t in self.ib.openTrades()]

    def place_moo(self, order: Order, ref: str) -> int:
        from ib_async import MarketOrder
        c = self._contracts([order.ticker])[order.ticker]
        o = MarketOrder("BUY" if order.qty > 0 else "SELL", abs(order.qty), tif="OPG",
                        orderRef=ref, account=self.account)
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


def _book_value(b: dict, prices: dict) -> float:
    return b["cash"] + sum(q * prices.get(t, float("nan")) for t, q in b["shares"].items())


def _apply(b: dict, ticker: str, shares: int, cash: float):
    b["shares"][ticker] = b["shares"].get(ticker, 0) + int(shares)
    if not b["shares"][ticker]:
        del b["shares"][ticker]
    b["cash"] += cash


def _fill_totals(fills_path: Path, order_ref: str) -> tuple:
    """(signed shares, average price, commission) of the recorded fills of one order."""
    if not fills_path.exists():
        return 0, float("nan"), 0.0
    rows = [r for r in csv.DictReader(fills_path.open()) if r.get("order_ref") == order_ref]
    sign = {"BOT": 1, "SLD": -1}
    shares = sum(sign.get(r["side"], 0) * float(r["shares"]) for r in rows)
    value = sum(float(r["shares"]) * float(r["price"]) for r in rows)
    comm = sum(float(r["commission"]) for r in rows if r.get("commission") not in ("", "nan", None))
    size = sum(float(r["shares"]) for r in rows)
    return int(round(shares)), (value / size if size else float("nan")), comm


def _default_target(book: Book, cfg: Config, broker, out: Path):
    from . import live
    if book.signal == MOMENTUM:
        def history(tickers, day):
            path = out / "prices" / f"closes_{day.date()}.parquet"
            if path.exists():
                return pd.read_parquet(path)
            closes = broker.history(tickers)
            path.parent.mkdir(parents=True, exist_ok=True)
            closes.to_parquet(path)
            return closes
        return lambda day: live.momentum_target(day, book.top_n, history=history, out=out)["tickers"]
    return lambda day: live.target(day, book.top_n, book.signal, book.largest, out=out)["tickers"]


def daily(cfg: Config, broker, day, submit=False, target_fn=None, calendar=None) -> str:
    """Record, decide, and (with `submit`) place the orders. Returns the message sent.
    `target_fn(book, day)` gives a book's target (default: `live`)."""
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
    books = state.setdefault("books", {})
    lines = [f"paper {day.date()}: account ${equity:,.0f} (cash ${cash:,.0f}), {len(held)} positions"]

    # fills of earlier decisions -> the books
    still = []
    for p in state.get("pending", []):
        if p["day"] >= str(day.date()):
            still.append(p)
            continue
        filled, price, comm = _fill_totals(out / "fills.csv", p["order_ref"])
        if filled != p["qty"]:
            lines.append(f"{p['ticker']}: {filled} of {p['qty']} shares filled")
        for b, (q, c) in allocate(p["alloc"], filled, price if filled else p["price"], comm).items():
            _apply(books[b], p["ticker"], q, c)
    state["pending"] = still

    # new books start with their capital in cash, if the account has it
    free = cash - sum(b["cash"] for b in books.values())
    for book in cfg.book_list():
        if book.name not in books:
            if free < book.capital * 0.98:
                raise RuntimeError(f"book {book.name}: ${book.capital:,.0f} wanted, ${free:,.0f} unallocated cash")
            books[book.name] = {"shares": {}, "cash": float(book.capital)}
            free -= book.capital
    owned = {}
    for b in books.values():
        for t, q in b["shares"].items():
            owned[t] = owned.get(t, 0) + q
    owned = {t: q for t, q in owned.items() if q}
    if owned != {t: q for t, q in held.items() if q}:
        raise RuntimeError(f"books do not add up to the account: books {owned}, account {held}")

    by_name = {b.name: b for b in cfg.book_list()}
    prices = broker.closes(sorted(owned)) if owned else {}
    rows = []
    for name, b in books.items():
        v = _book_value(b, prices)
        rows.append({"day": str(day.date()), "book": name, "equity_usd": round(v, 2), "cash_usd": round(b["cash"], 2),
                     "positions": json.dumps(b["shares"], sort_keys=True)})
        lines.append(f"  {name}: ${v:,.0f} (cash ${b['cash']:,.0f}), {len(b['shares'])} positions")
    _append_csv(out / "books.csv", rows)
    lines.append(f"  unallocated cash ${cash - sum(b['cash'] for b in books.values()):,.0f}")

    book_orders, decided = {}, []
    for name, book in by_name.items():
        b = books[name]
        due = "target" not in b or is_period_end(day, book.every, calendar)
        if b.get("last_decision") == str(day.date()) or not due:
            continue
        fn = target_fn or (lambda bk, d: _default_target(bk, cfg, broker, out)(d))
        target = fn(book, day)
        decided.append(name)
        lines.append(f"{name} target ({book.signal} top {book.top_n}): {', '.join(target)}")
        b["new_target"] = target
        if not needs_rebalance(target, b["shares"]):
            lines.append(f"{name}: same names as held, no trades")
            continue
        need = sorted(set(target) | set(b["shares"]) - set(prices))
        prices.update(broker.closes(need) if need else {})
        orders = plan(target, b["shares"], prices, _book_value(b, prices), cfg.cash_buffer, cfg.min_order_value)
        problems = check(orders, _book_value(b, prices), cfg, book.top_n)
        if problems:
            raise RuntimeError(f"{name} plan refused: " + "; ".join(problems))
        book_orders[name] = orders
    orders, alloc = net_orders(book_orders)
    problems = []
    if orders and equity > cfg.max_equity:
        problems.append(f"account worth ${equity:,.0f} > max_equity ${cfg.max_equity:,.0f}: reset the paper balance")
    if len(orders) > cfg.max_orders:
        problems.append(f"{len(orders)} orders > max_orders {cfg.max_orders}")
    if problems:
        raise RuntimeError("plan refused: " + "; ".join(problems))
    if orders or alloc:
        opened = broker.open_orders()
        if opened:
            raise RuntimeError(f"orders still open, refusing to add more: {opened}")
    for o in orders:
        parts = ", ".join(f"{b} {q:+d}" for b, q in sorted(alloc[o.ticker].items()))
        lines.append(f"  {'BUY ' if o.qty > 0 else 'SELL'} {abs(o.qty):5d} {o.ticker:6s} ~${o.price:,.2f} ({parts})")
    crossed = [t for t, a in alloc.items() if not sum(a.values())]
    for t in crossed:
        lines.append(f"  {t}: books trade {alloc[t]} with each other at ${prices[t]:,.2f}, no order")
    if submit:
        ids = [broker.place_moo(o, ref=f"algotrader:{day.date()}:{o.ticker}") for o in orders]
        _append_csv(out / "orders.csv", [dict(asdict(o), day=str(day.date()), order_id=i) for o, i in zip(orders, ids)])
        for o in orders:
            state["pending"].append({"day": str(day.date()), "ticker": o.ticker, "qty": o.qty, "price": o.price,
                                     "order_ref": f"algotrader:{day.date()}:{o.ticker}", "alloc": alloc[o.ticker]})
        for t in crossed:
            for b, (q, c) in allocate(alloc[t], 0, prices[t]).items():
                _apply(books[b], t, q, c)
        for name in decided:
            books[name]["target"] = books[name].pop("new_target")
            books[name]["last_decision"] = str(day.date())
        state_path.write_text(json.dumps(state, indent=2))
        lines.append(f"{len(orders)} market-on-open orders placed" if orders else "nothing to place")
    elif decided:
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
            st = json.loads(state.read_text()) if state.exists() else {}
            for name, b in st.get("books", {}).items():
                print(f"book {name}: cash ${b['cash']:,.0f}, shares {b['shares']}, target {b.get('target')}, "
                      f"last decision {b.get('last_decision')}")
            print("pending:", st.get("pending", []) or "none")
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
