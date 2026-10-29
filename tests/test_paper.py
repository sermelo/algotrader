import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from algotrader import live, paper
from algotrader.paper import Config, Order, check, daily, from_ib_symbol, ib_symbol, is_month_end, needs_rebalance, plan

MEMBERS_HTML = """<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>CIK</th></tr>
<tr><td>AAA</td><td>Alpha</td><td>Tech</td><td>0000000001</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Fin</td><td>0001067983</td></tr>
""" + "".join(f"<tr><td>X{i}</td><td>X{i} Inc</td><td>Tech</td><td>{100 + i:010d}</td></tr>" for i in range(500)) + "</table>"


def test_members_table_parsed_with_cik():
    m = live.parse_members(MEMBERS_HTML)
    assert len(m) == 502 and m.loc["BRK.B", "cik"] == 1067983 and m.loc["AAA", "name"] == "Alpha"
    with pytest.raises(ValueError):
        live.parse_members(MEMBERS_HTML[:400] + "</table>")        # too few rows: a changed page


def facts_rows(cik, filed, op, eq, end="2025-12-31"):
    base = dict(cik=cik, taxonomy="us-gaap", unit="USD", start=pd.Timestamp(end) - pd.Timedelta(days=364),
                end=pd.Timestamp(end), filed=pd.Timestamp(filed), form="10-K", fy=2025, fp="FY", accession="x")
    return [dict(base, tag="OperatingIncomeLoss", value=op), dict(base, tag="StockholdersEquity", value=eq)]


def test_scores_use_only_facts_filed_before_the_day_and_choose_ranks():
    members = pd.DataFrame({"cik": [1, 2, 3], "name": list("abc"), "sector": "x"}, index=["A", "B", "C"])
    facts = pd.DataFrame(facts_rows(1, "2026-02-01", 10, 100) + facts_rows(2, "2026-02-01", 30, 100)
                         + facts_rows(3, "2026-03-10", 90, 100) + facts_rows(3, "2026-02-01", 5, -10, end="2024-12-31"))
    s = live.scores(facts, members, "2026-03-10")          # C's report filed that day: not yet usable
    assert s["A"] == pytest.approx(0.1) and s["B"] == pytest.approx(0.3) and pd.isna(s["C"])
    s2 = live.scores(facts, members, "2026-03-11")
    assert s2["C"] == pytest.approx(0.9)
    assert live.choose(s2, 2) == ["C", "B"] and live.choose(s2, 1, largest=False) == ["A"]
    assert live.choose(pd.Series([1.0, 1.0, 0.5], index=["Y", "Z", "W"]), 1) == ["Y"]   # tie: first in order
    with pytest.raises(ValueError):
        live.choose(s2, 0)


def test_target_writes_its_file(tmp_path):
    members = pd.DataFrame({"cik": [1, 2], "name": ["a", "b"], "sector": "x"}, index=["A", "B"])
    facts = pd.DataFrame(facts_rows(1, "2026-02-01", 10, 100) + facts_rows(2, "2026-02-01", 30, 100))
    res = live.target("2026-03-31", 1, out=tmp_path, members=members, facts=facts)
    assert res["tickers"] == ["B"] and json.loads((tmp_path / "target_operating_profitability_2026-03-31.json").read_text())["tickers"] == ["B"]


def test_month_end_sessions():
    assert is_month_end("2026-10-30")                 # Friday, last session of October
    assert not is_month_end("2026-10-29") and not is_month_end("2026-10-31")   # Saturday
    assert is_month_end("2026-12-31") and is_month_end("2027-04-30")
    assert not is_month_end("2026-11-26")             # Thanksgiving, no session
    assert is_month_end("2027-05-28")                 # the 31st is Memorial Day


def test_rebalance_only_when_names_change():
    assert not needs_rebalance(["A", "B"], {"B": 3, "A": 1})
    assert needs_rebalance(["A", "B"], {"A": 1, "C": 2}) and needs_rebalance(["A"], {"A": 1, "B": 0, "C": 1})
    assert not needs_rebalance(["A"], {"A": 5, "B": 0})


def test_plan_equal_weight_whole_shares_sells_first():
    orders = plan(["A", "B"], {"B": 10, "C": 7}, {"A": 33.0, "B": 50.0, "C": 20.0}, equity=10_000, cash_buffer=0.01)
    assert [(o.ticker, o.qty, o.reason) for o in orders] == [("C", -7, "exit"), ("A", 150, "enter"), ("B", 89, "rebalance")]
    # 4,950 per slot: floor(4950/33) = 150, floor(4950/50) = 99 -> buy 89 more
    small = plan(["A"], {"A": 140}, {"A": 33.0}, equity=4_800, cash_buffer=0.0, min_order_value=200)
    assert small == []                                # 145 wanted, 5 x $33 below $200
    assert plan([], {"A": 3}, {"A": 1.0}, 100) == [Order("A", -3, 1.0, "exit")]
    with pytest.raises(ValueError):
        plan(["A", "B"], {}, {"A": 10.0}, 1000)
    pricey = plan(["A", "B"], {}, {"A": 3000.0, "B": 10.0}, equity=5000)   # a share over the slot is not bought
    assert [o.ticker for o in pricey] == ["B"]


def test_check_refusals():
    cfg = Config(capital=15_000, max_equity=30_000, max_orders=2, top_n=10)
    assert check([Order("A", 20, 100.0, "enter")], 15_000, cfg) == []     # $2,000 <= 1.5 x $1,500
    assert any("reset the paper balance" in p for p in check([], 1_000_000, cfg))
    assert any("max_orders" in p for p in check([Order("A", 1, 1.0, "x")] * 3, 15_000, cfg))
    assert any("slots" in p for p in check([Order("A", 23, 100.0, "enter")], 15_000, cfg))


def test_config_validation_and_file():
    cfg = Config.load("config/paper.json")
    assert cfg.mode == "paper" and cfg.top_n == 10 and cfg.capital == 15_000
    for kw in ({"mode": "x"}, {"top_n": 0}, {"capital": 0}, {"max_equity": 1}, {"max_order_slots": 0.5}):
        with pytest.raises(ValueError):
            Config(**kw)


def test_symbols():
    assert ib_symbol("BRK.B") == "BRK B" and from_ib_symbol("BRK B") == "BRK.B" and ib_symbol("AAPL") == "AAPL"


class FakeBroker:
    def __init__(self, account="DU123", equity=15_000.0, held=None, prices=None, open_=(), cash=None, fills=()):
        self.account, self.equity, self.held = account, equity, dict(held or {})
        self.prices = prices or {}
        self.cash = cash
        self.open_, self.placed, self.refs = list(open_), [], []
        self.fills = list(fills) or [{"exec_id": "e1", "ticker": "A", "shares": 1.0}]

    def equity_usd(self): return self.equity
    def fx(self): return {"base_currency": "USD", "equity_base": self.equity, "usd_per_base": 1.0, "rate_source": "base"}
    def cash_usd(self):
        return self.cash if self.cash is not None else self.equity - sum(q * self.prices.get(t, 0) for t, q in self.held.items())
    def positions(self): return dict(self.held)
    def closes(self, tickers): return {t: self.prices[t] for t in tickers if t in self.prices}
    def open_orders(self): return self.open_
    def place_moo(self, o, ref): self.placed.append(o); self.refs.append(ref); return len(self.placed)
    def fills_today(self): return self.fills


def cfg_in(tmp_path, **kw):
    return Config(out=str(tmp_path), **kw)


def fixed(*names):
    return lambda book, day: list(names)


def main_book(tmp_path):
    return json.loads((tmp_path / "state.json").read_text())["books"]["main"]


def test_daily_first_run_builds_then_waits_for_month_end(tmp_path):
    cfg = cfg_in(tmp_path, top_n=2)
    b = FakeBroker(prices={"A": 50.0, "B": 100.0})
    msg = daily(cfg, b, "2026-10-22", submit=True, target_fn=fixed("A", "B"))
    assert [(o.ticker, o.qty) for o in b.placed] == [("A", 148), ("B", 74)] and "2 market-on-open" in msg
    assert b.refs == ["algotrader:2026-10-22:A", "algotrader:2026-10-22:B"]
    assert main_book(tmp_path)["target"] == ["A", "B"] and main_book(tmp_path)["shares"] == {}
    assert (tmp_path / "orders.csv").exists() and (tmp_path / "equity.csv").exists()
    fills = [{"exec_id": "x1", "ticker": "A", "side": "BOT", "shares": 148.0, "price": 51.0, "commission": 1.0,
              "order_ref": "algotrader:2026-10-22:A"},
             {"exec_id": "x2", "ticker": "B", "side": "BOT", "shares": 74.0, "price": 99.0, "commission": 1.0,
              "order_ref": "algotrader:2026-10-22:B"}]
    b2 = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0}, fills=fills)
    called = []
    daily(cfg, b2, "2026-10-23", submit=True, target_fn=lambda bk, d: called.append(d) or ["A", "C"])
    assert called == [] and b2.placed == []          # not a month end: only recorded
    book = main_book(tmp_path)
    assert book["shares"] == {"A": 148, "B": 74} and book["cash"] == pytest.approx(15_000 - 148 * 51 - 74 * 99 - 2)
    assert json.loads((tmp_path / "state.json").read_text())["pending"] == []
    rows = (tmp_path / "equity.csv").read_text().splitlines()
    assert len(rows) == 3 and len((tmp_path / "fills.csv").read_text().splitlines()) == 4
    assert len((tmp_path / "books.csv").read_text().splitlines()) == 3


def state_with(tmp_path, shares, cash, target, name="main"):
    (tmp_path / "state.json").write_text(json.dumps(
        {"books": {name: {"shares": shares, "cash": cash, "target": target}}, "pending": []}))


def test_daily_month_end_trades_only_when_names_change(tmp_path):
    cfg = cfg_in(tmp_path, top_n=2)
    state_with(tmp_path, {"A": 148, "B": 74}, 200.0, ["A", "B"])
    b = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0, "C": 10.0})
    assert "no trades" in daily(cfg, b, "2026-10-30", submit=True, target_fn=fixed("B", "A")) and b.placed == []
    state_with(tmp_path, {"A": 148, "B": 74}, 200.0, ["A", "B"])
    b = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0, "C": 10.0})
    daily(cfg, b, "2026-11-30", submit=True, target_fn=fixed("A", "C"))
    # book worth 200 + 7,400 + 7,400 = 15,000; x 0.99 / 2 = 7,425 a slot: A wants 148 (held), C 742; B sold first
    assert [(o.ticker, o.qty) for o in b.placed] == [("B", -74), ("C", 742)]
    again = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0, "C": 10.0})
    daily(cfg, again, "2026-11-30", submit=True, target_fn=fixed("A", "D"))
    assert again.placed == []                         # one decision a day


def test_daily_safety(tmp_path):
    cfg = cfg_in(tmp_path, top_n=1)
    t = fixed("A")
    with pytest.raises(RuntimeError, match="does not match"):
        daily(cfg, FakeBroker(account="U999"), "2026-10-22", True, t)
    with pytest.raises(RuntimeError, match="reset the paper balance"):
        daily(cfg, FakeBroker(equity=1_000_000, prices={"A": 10.0}), "2026-10-22", True, t)
    with pytest.raises(RuntimeError, match="still open"):
        daily(cfg, FakeBroker(prices={"A": 10.0}, open_=["BUY 1 A"]), "2026-10-22", True, t)
    with pytest.raises(ValueError, match="no price"):
        daily(cfg, FakeBroker(prices={}), "2026-10-22", True, t)
    with pytest.raises(RuntimeError, match="unallocated cash"):
        daily(cfg, FakeBroker(equity=5_000, prices={"A": 10.0}), "2026-10-22", True, t)
    with pytest.raises(RuntimeError, match="do not add up"):    # a position no book owns
        daily(cfg, FakeBroker(held={"Z": 3}, prices={"A": 10.0, "Z": 1.0}), "2026-10-22", True, t)
    dry = FakeBroker(prices={"A": 10.0})
    assert "dry run" in daily(cfg, dry, "2026-10-22", False, t) and dry.placed == []
    assert not (tmp_path / "state.json").exists()
    (tmp_path / "STOP").touch()
    stopped = FakeBroker(prices={"A": 10.0})
    assert "STOP" in daily(cfg, stopped, "2026-10-22", True, t) and stopped.placed == []


def test_week_ends():
    assert paper.is_period_end("2026-10-30", "W") and not paper.is_period_end("2026-10-29", "W")
    assert paper.is_period_end("2026-04-02", "W")      # Good Friday 2026-04-03: Thursday ends the week
    assert not paper.is_period_end("2026-10-31", "W")  # Saturday
    assert paper.is_period_end("2026-10-30", "M") and not paper.is_period_end("2026-11-27", "M")


def test_net_orders_and_allocation():
    orders = {"op": [Order("A", 10, 50.0, "enter"), Order("B", -4, 20.0, "exit")],
              "mom": [Order("A", -4, 50.0, "exit"), Order("B", 4, 20.0, "enter"), Order("C", 3, 9.0, "enter")]}
    net, alloc = paper.net_orders(orders)
    assert [(o.ticker, o.qty) for o in net] == [("A", 6), ("C", 3)]        # B cancels out: no order
    assert alloc["A"] == {"op": 10, "mom": -4} and alloc["B"] == {"op": -4, "mom": 4}
    full = paper.allocate(alloc["A"], 6, 51.0, 1.2)        # mom sells 4 to op, the market sells op 6 more
    assert full == {"mom": (-4, 4 * 51.0), "op": (10, -10 * 51.0 - 1.2)}
    part = paper.allocate(alloc["A"], 3, 51.0, 1.0)        # only 3 of 6 filled: op gets 4 + 3
    assert part["op"][0] == 7 and part["mom"][0] == -4
    assert sum(q for q, _ in part.values()) == 3 and sum(c for _, c in part.values()) == pytest.approx(-3 * 51 - 1)
    cross = paper.allocate(alloc["B"], 0, 20.0)
    assert cross == {"op": (-4, 80.0), "mom": (4, -80.0)}
    two = paper.allocate({"x": 3, "y": 1, "z": -2}, 1, 10.0, 2.0)  # 1 filled + 2 crossed = 3 for x and y
    assert two["x"][0] + two["y"][0] == 3 and two["z"] == (-2, 20.0)
    assert sum(q for q, _ in two.values()) == 1 and sum(c for _, c in two.values()) == pytest.approx(-12.0)


def two_books(tmp_path):
    return cfg_in(tmp_path, max_equity=45_000, books=(paper.Book("op", "operating_profitability", 2, "M", 15_000),
                                                       paper.Book("mom", "momentum", 1, "W", 15_000)))


def test_two_books_net_their_orders_and_split_the_fills(tmp_path):
    cfg = two_books(tmp_path)
    targets = {"op": ["A", "B"], "mom": ["A"]}
    b = FakeBroker(equity=30_000, prices={"A": 50.0, "B": 100.0})
    msg = daily(cfg, b, "2026-10-22", True, lambda bk, d: targets[bk.name])
    # op: 7,425 / 50 = 148 A, 74 B; mom: 14,850 / 50 = 297 A -> one order of 445 A
    assert [(o.ticker, o.qty) for o in b.placed] == [("A", 445), ("B", 74)] and "mom +297, op +148" in msg
    fills = [{"exec_id": "f1", "ticker": "A", "side": "BOT", "shares": 445.0, "price": 50.0, "commission": 4.45,
              "order_ref": "algotrader:2026-10-22:A"},
             {"exec_id": "f2", "ticker": "B", "side": "BOT", "shares": 74.0, "price": 100.0, "commission": 1.0,
              "order_ref": "algotrader:2026-10-22:B"}]
    b2 = FakeBroker(equity=30_000, held={"A": 445, "B": 74}, prices={"A": 50.0, "B": 100.0}, fills=fills)
    msg = daily(cfg, b2, "2026-10-28", True, lambda bk, d: pytest.fail("not a decision day"))
    st = json.loads((tmp_path / "state.json").read_text())["books"]
    assert st["op"]["shares"] == {"A": 148, "B": 74} and st["mom"]["shares"] == {"A": 297}
    assert st["op"]["cash"] == pytest.approx(15_000 - 7_400 - 7_400 - 4.45 * 148 / 445 - 1.0)
    assert st["mom"]["cash"] == pytest.approx(15_000 - 14_850 - 4.45 * 297 / 445)
    # Friday 2026-10-30 is a week end and a month end: mom moves from A to B, op from B to C
    saved = (tmp_path / "state.json").read_text()
    targets = {"op": ["A", "C"], "mom": ["B"]}
    b3 = FakeBroker(equity=30_000, held={"A": 445, "B": 74}, prices={"A": 50.0, "B": 100.0, "C": 25.0})
    daily(cfg, b3, "2026-10-30", True, lambda bk, d: targets[bk.name])
    # op sells its 74 B to mom, mom buys 148 - 74 more in the market; op buys 296 C; mom sells its 297 A
    assert [(o.ticker, o.qty) for o in b3.placed] == [("A", -297), ("B", 74), ("C", 296)]
    st = json.loads((tmp_path / "state.json").read_text())
    assert {p["ticker"]: p["alloc"] for p in st["pending"]} == {"A": {"mom": -297}, "B": {"op": -74, "mom": 148},
                                                                "C": {"op": 296}}
    assert st["books"]["mom"]["last_decision"] == st["books"]["op"]["last_decision"] == "2026-10-30"
    # at B = $200 mom wants exactly op's 74 B: the books trade them at the close, no order
    (tmp_path / "state.json").write_text(saved)
    b4 = FakeBroker(equity=30_000, held={"A": 445, "B": 74}, prices={"A": 50.0, "B": 200.0, "C": 25.0})
    msg = daily(cfg, b4, "2026-10-30", True, lambda bk, d: targets[bk.name])
    assert "B" not in {o.ticker for o in b4.placed} and "no order" in msg
    books = json.loads((tmp_path / "state.json").read_text())["books"]
    before = json.loads(saved)["books"]
    assert "B" not in books["op"]["shares"] and books["mom"]["shares"]["B"] == 74
    assert books["op"]["cash"] == pytest.approx(before["op"]["cash"] + 74 * 200)
    assert books["mom"]["cash"] == pytest.approx(before["mom"]["cash"] - 74 * 200)


def test_weekly_book_decides_only_at_week_end(tmp_path):
    cfg = two_books(tmp_path)
    (tmp_path / "state.json").write_text(json.dumps({"books": {
        "op": {"shares": {}, "cash": 15_000.0, "target": []}, "mom": {"shares": {}, "cash": 15_000.0, "target": []}},
        "pending": []}))
    asked = []
    daily(cfg, FakeBroker(equity=30_000), "2026-10-28", True, lambda bk, d: asked.append(bk.name) or [])
    daily(cfg, FakeBroker(equity=30_000), "2026-11-06", True, lambda bk, d: asked.append(bk.name) or [])
    assert asked == ["mom"]                            # Friday 2026-11-06: week end, not month end


def test_notify_runs_the_command(tmp_path):
    log = tmp_path / "log.txt"
    cfg = Config(notify_cmd=["sh", "-c", f'echo "$0" > {log}'])
    paper.notify(cfg, "hello")
    assert log.read_text().strip() == "hello"


ECB_XML = """<gesmes:Envelope><Cube><Cube time='2026-10-27'><Cube currency='USD' rate='1.1612'/>
<Cube currency='JPY' rate='172.10'/></Cube></Cube></gesmes:Envelope>"""


def test_usd_per_base_currency():
    usd = {("NetLiquidation", "USD"): "15000"}
    assert paper.usd_per_base(usd) == 1.0
    held = {("NetLiquidation", "EUR"): "13000", ("ExchangeRate", "USD"): "0.8"}   # 0.8 EUR per USD
    assert paper.usd_per_base(held, ecb=lambda: 9.9) == pytest.approx(1.25)
    ledger = {("NetLiquidation", "EUR"): "13107.88", ("$LEDGER-ExchangeRate", "USD"): "0.8970318"}  # paper account
    assert paper.usd_rate(ledger, ecb=lambda: 9.9) == (pytest.approx(1.114788, abs=1e-6), "ibkr")
    fresh = {("NetLiquidation", "EUR"): "13000", ("ExchangeRate", "EUR"): "1.00"}  # no dollars held yet
    assert paper.usd_rate(fresh, ecb=lambda: 1.1612) == (1.1612, "ecb")
    assert paper.usd_per_base(fresh, ecb=lambda: 1.1612) == 1.1612
    with pytest.raises(RuntimeError):                                              # never assume 1
        paper.usd_per_base({("NetLiquidation", "GBP"): "10000"}, ecb=lambda: 1.16)
    assert paper.parse_ecb(ECB_XML) == {"USD": 1.1612, "JPY": 172.10}
    with pytest.raises(ValueError):
        paper.parse_ecb("<Cube currency='JPY' rate='172.1'/>")


def test_equity_record_gets_new_columns_without_losing_rows(tmp_path):
    (tmp_path / "equity.csv").write_text("day,equity_usd,cash_usd,positions\n2026-10-27,14665.75,14544.4,{}\n")
    daily(cfg_in(tmp_path), FakeBroker(), "2026-10-28", target_fn=fixed())
    rows = list(csv.DictReader((tmp_path / "equity.csv").open()))
    assert [r["day"] for r in rows] == ["2026-10-27", "2026-10-28"]
    assert rows[0]["usd_per_base"] == "" and rows[1]["usd_per_base"] == "1.0" and rows[1]["rate_source"] == "base"
    daily(cfg_in(tmp_path), FakeBroker(), "2026-10-29", target_fn=fixed())
    assert (tmp_path / "equity.csv").read_text().count("day,") == 1


def momentum_closes(days=300):
    idx = pd.bdate_range("2025-08-01", periods=days)
    up = pd.Series(range(days), index=idx, dtype=float)
    return pd.DataFrame({"AAA": 100 + up, "BBB": 100 + 0.5 * up, "CCC": 100.0 + 0 * up}, index=idx)


def test_momentum_scores_are_the_backtests():
    closes = momentum_closes()
    s = live.momentum_scores(closes)
    assert s["AAA"] == pytest.approx(closes["AAA"].iloc[-22] / closes["AAA"].iloc[-253] - 1)
    wrecked = closes.copy()
    wrecked.iloc[-21:] *= 3                             # the last month is skipped
    assert live.momentum_scores(wrecked)["AAA"] == pytest.approx(s["AAA"])
    gone = closes.copy()
    gone.loc[gone.index[-1], "AAA"] = float("nan")       # no close on the day: not ranked (backtest)
    assert pd.isna(live.momentum_scores(gone)["AAA"])
    from algotrader.portfolio import CrossSectionalMomentum
    pd.testing.assert_series_equal(s, CrossSectionalMomentum().score(closes).iloc[-1], check_names=False)


def test_momentum_target(tmp_path):
    closes = momentum_closes()
    day = closes.index[-1]
    members = pd.DataFrame({"cik": [1, 2, 3], "name": ["a", "b", "c"], "sector": "x"}, index=["CCC", "BBB", "AAA"])
    asked = []
    res = live.momentum_target(day, 2, history=lambda t, d: asked.append((t, d)) or closes, out=tmp_path,
                               members=members)
    assert res["tickers"] == ["AAA", "BBB"] and asked == [(["CCC", "BBB", "AAA"], day)]
    assert json.loads((tmp_path / f"target_momentum_{day.date()}.json").read_text())["scored"] == 3
    with pytest.raises(ValueError, match="not on"):
        live.momentum_target(day + pd.Timedelta(days=3), 2, out=tmp_path, members=members, closes=closes)
    with pytest.raises(ValueError, match="have a close"):
        live.momentum_target(day, 2, out=tmp_path, members=members, closes=closes.drop(columns=["AAA"]))


def test_config_with_books(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_equity": 45000, "books": [
        {"name": "op", "signal": "operating_profitability", "top_n": 10, "every": "M", "capital": 15000},
        {"name": "mom", "signal": "momentum", "top_n": 5, "every": "W", "capital": 15000}]}))
    cfg = Config.load(p)
    assert [b.name for b in cfg.book_list()] == ["op", "mom"] and cfg.books[1].every == "W"
    assert Config().book_list()[0].name == "main"
    with pytest.raises(ValueError):
        Config(books=(paper.Book("x"), paper.Book("x")))
    with pytest.raises(ValueError):
        paper.Book("x", every="D")
    assert Config.load(Path("config/paper.json")).book_list()
