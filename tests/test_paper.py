import json

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
    assert res["tickers"] == ["B"] and json.loads((tmp_path / "target_2026-03-31.json").read_text())["tickers"] == ["B"]


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
    def __init__(self, account="DU123", equity=15_000.0, held=None, prices=None, open_=()):
        self.account, self.equity, self.held = account, equity, dict(held or {})
        self.prices = prices or {}
        self.open_, self.placed = list(open_), []

    def equity_usd(self): return self.equity
    def cash_usd(self): return self.equity - sum(q * self.prices.get(t, 0) for t, q in self.held.items())
    def positions(self): return dict(self.held)
    def closes(self, tickers): return {t: self.prices[t] for t in tickers if t in self.prices}
    def open_orders(self): return self.open_
    def place_moo(self, o): self.placed.append(o); return len(self.placed)
    def fills_today(self): return [{"exec_id": "e1", "ticker": "A", "shares": 1.0}]


def cfg_in(tmp_path, **kw):
    return Config(out=str(tmp_path), **kw)


def test_daily_first_run_builds_then_waits_for_month_end(tmp_path):
    cfg = cfg_in(tmp_path, top_n=2)
    b = FakeBroker(prices={"A": 50.0, "B": 100.0})
    msg = daily(cfg, b, "2026-10-22", submit=True, target_fn=lambda d: ["A", "B"])
    assert [(o.ticker, o.qty) for o in b.placed] == [("A", 148), ("B", 74)] and "2 market-on-open" in msg
    assert json.loads((tmp_path / "state.json").read_text())["target"] == ["A", "B"]
    assert (tmp_path / "orders.csv").exists() and (tmp_path / "equity.csv").exists()
    b2 = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0})
    called = []
    daily(cfg, b2, "2026-10-23", submit=True, target_fn=lambda d: called.append(d) or ["A", "C"])
    assert called == [] and b2.placed == []          # not a month end: only recorded
    rows = (tmp_path / "equity.csv").read_text().splitlines()
    assert len(rows) == 3 and len((tmp_path / "fills.csv").read_text().splitlines()) == 2   # fill e1 once


def test_daily_month_end_trades_only_when_names_change(tmp_path):
    cfg = cfg_in(tmp_path, top_n=2)
    (tmp_path / "state.json").write_text(json.dumps({"target": ["A", "B"]}))
    b = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0, "C": 10.0})
    assert "no trades" in daily(cfg, b, "2026-10-30", submit=True, target_fn=lambda d: ["B", "A"]) and b.placed == []
    b = FakeBroker(held={"A": 148, "B": 74}, prices={"A": 50.0, "B": 100.0, "C": 10.0})
    daily(cfg, b, "2026-11-30", submit=True, target_fn=lambda d: ["A", "C"])
    # 15,000 x 0.99 / 2 = 7,425 a slot: A wants 148 (held), C 742; B sold first
    assert [(o.ticker, o.qty) for o in b.placed] == [("B", -74), ("C", 742)]
    again = FakeBroker(held={"A": 148, "C": 742}, prices={"A": 50.0, "C": 10.0})
    daily(cfg, again, "2026-11-30", submit=True, target_fn=lambda d: ["A", "D"])
    assert again.placed == []                         # one decision a day


def test_daily_safety(tmp_path):
    cfg = cfg_in(tmp_path, top_n=1)
    t = lambda d: ["A"]
    with pytest.raises(RuntimeError, match="does not match"):
        daily(cfg, FakeBroker(account="U999"), "2026-10-22", True, t)
    with pytest.raises(RuntimeError, match="reset the paper balance"):
        daily(cfg, FakeBroker(equity=1_000_000, prices={"A": 10.0}), "2026-10-22", True, t)
    with pytest.raises(RuntimeError, match="still open"):
        daily(cfg, FakeBroker(prices={"A": 10.0}, open_=["BUY 1 A"]), "2026-10-22", True, t)
    with pytest.raises(ValueError, match="no price"):
        daily(cfg, FakeBroker(prices={}), "2026-10-22", True, t)
    dry = FakeBroker(prices={"A": 10.0})
    assert "dry run" in daily(cfg, dry, "2026-10-22", False, t) and dry.placed == []
    assert not (tmp_path / "state.json").exists()
    (tmp_path / "STOP").touch()
    stopped = FakeBroker(prices={"A": 10.0})
    assert "STOP" in daily(cfg, stopped, "2026-10-22", True, t) and stopped.placed == []


def test_notify_runs_the_command(tmp_path):
    log = tmp_path / "log.txt"
    cfg = Config(notify_cmd=["sh", "-c", f'echo "$0" > {log}'])
    paper.notify(cfg, "hello")
    assert log.read_text().strip() == "hello"
