"""Merger arbitrage on announced cash takeovers (roadmap S15, 2026-10-20).

A risk premium, not a price pattern: after a cash takeover is announced the target trades a few percent
below the offer, because some deals fail and the target then falls back. Buying the targets earns that
spread on the deals that close and loses on the ones that fail (Mitchell & Pulvino, *JF* 2001: about 4%
a year over cash, beta near 0 in normal markets and about 0.5 in falling ones).

Deals come from SEC filings (free bulk index `submissions.zip`, see `edgar.py`): the target's
preliminary or definitive merger proxy or information statement (PREM14A, DEFM14A, PREM14C, DEFM14C) or a
tender-offer filing (SC TO-T, SC TO-C, SC 14D9, SC14D9C). Each filing's text gives the ticker candidates
("Nasdaq: ABCD", "under the symbol "ABCD"") and the cash price per share ("$12.50 in cash", "$12.50 per
Share, net to the seller in cash"). A filing becomes a deal on the first trading day the market could react
to it (accepted at or after 16:00 New York: the next day), if exactly one candidate ticker closes that day
below the offer by at most `max_spread` (so the ticker is the target and the spread is known at the
decision), the close is at least `min_price` and its 20-day median dollar volume at least `min_dollar_volume`.
A ticker has at most one deal at a time; a filing for a ticker already in a deal (or within 365 days of its
previous deal start) is ignored. Blank-check companies (SIC 6770) are skipped.

Holding: from the next open until the price series ends (the deal closed; sold at the last close), until
the next open after any of the deal's companies files an 8-K with item 1.02 (termination of a material
agreement) or after `max_days` calendar days. At most `slots` deals at once, each 1/`slots` of equity
(first come; a deal arriving when all slots are full is skipped); cash earns the risk-free rate (Fama-French
daily RF), added after the backtest.

Look-ahead: the deal list uses only each filing's own text and the prices up to its decision day; which
deals later close is never used to select them. Entering at the proxy or tender filing (weeks after the
announcement for mergers) misses the first jump; that is the honest cost of a rule built only from filings.

    python -m algotrader.mergerarb fetch      # VPS: download and parse the filings -> data/merger/filings.parquet
    python -m algotrader.mergerarb deals      # match to prices -> data/merger/deals.csv
    python -m algotrader.mergerarb run        # backtest -> results/merger/
"""
import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .costs import CostModel
from .pead import reaction_days
from .portfolio import PortfolioBacktester

EDGAR = Path("data/edgar")
DATA = Path("data/merger")
HF_DAILY = Path("data/hf_daily")
OUT = Path("results/merger")
FORMS = ("PREM14A", "DEFM14A", "PREM14C", "DEFM14C", "SC TO-T", "SC TO-C", "SC 14D9", "SC14D9C")
START, FIRST_TEST = "2008-01-01", "2009-01-01"
CAPITAL = 1_000_000.0
COSTS = {"base": CostModel(spread_bps=5.0), "gross": CostModel(0.0, 0.0, 0.0, 0.0),
         "stress": CostModel(spread_bps=10.0)}
READ_BYTES = 4_000_000                       # the offer terms are in the first pages
BLANK_CHECK = "6770"

_EXCH = r"(?:NYSE American|NYSE MKT|NYSE Amex|NYSE Arca|NYSE|NASDAQ|Nasdaq|AMEX)"
TICKER = re.compile(_EXCH + r"[A-Za-z ]{0,20}:\s*([A-Z]{1,5})\b|"
                    r"symbols?\s*[“\"']([A-Z]{1,5})[”\"'.,]")
CASH = re.compile(r"\$\s?(\d{1,4}(?:,\d{3})*(?:\.\d{1,4})?)\s+(?:per share\s+)?in cash|"
                  r"\$\s?(\d{1,4}(?:,\d{3})*(?:\.\d{1,4})?)\s+per\s+(?:share|Share)[^.$]{0,60}?"
                  r"(?:in cash|net to the seller)")


# ---------------------------------------------------------------- parsing

def clean_text(raw: bytes) -> str:
    """Filing bytes to plain text: tags removed, entities decoded, single spaces."""
    t = raw.decode("utf-8", "ignore")
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def tickers(text: str) -> list:
    """Ticker candidates named in the text, sorted."""
    return sorted({a or b for a, b in TICKER.findall(text)})


def cash_price(text: str) -> float:
    """The cash price per share named most often (ties: the first named); NaN if none."""
    found = [float((a or b).replace(",", "")) for a, b in CASH.findall(text)]
    found = [x for x in found if x > 0]
    if not found:
        return float("nan")
    counts = pd.Series(found).value_counts(sort=False)
    return float(counts.index[counts.to_numpy().argmax()])


# ---------------------------------------------------------------- download (VPS)

def filing_list(edgar=EDGAR, start=START) -> pd.DataFrame:
    """The deal filings (FORMS) from `deal_filings.parquet` (`scripts/deal_forms.py`), from `start`."""
    df = pd.read_parquet(Path(edgar) / "deal_filings.parquet")
    return df[df["form"].isin(FORMS) & (df["filed"] >= start)].sort_values(["filed", "accession"])


def fetch(out=DATA, edgar=EDGAR, opener=urllib.request.urlopen, workers=4, pause=0.4, ua_file=None):
    """Download each filing's full text (first READ_BYTES), keep tickers and cash price only.
    `workers` threads, each pausing `pause` s per request (SEC allows 10 requests a second).
    Resumable: filings already in `filings.parquet` are skipped (saved every 200)."""
    from concurrent.futures import ThreadPoolExecutor

    from .edgar import UA_FILE, user_agent
    ua = user_agent(ua_file or UA_FILE)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "filings.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    seen = set(done["accession"] + "|" + done["cik"].astype(str)) if len(done) else set()
    todo = filing_list(edgar)
    todo = todo[~(todo["accession"] + "|" + todo["cik"].astype(str)).isin(seen)]

    def one(r):
        url = f"https://www.sec.gov/Archives/edgar/data/{r.cik}/{r.accession}.txt"
        try:
            with opener(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=60) as f:
                text = clean_text(f.read(READ_BYTES))
            return {"cik": r.cik, "accession": r.accession, "form": r.form, "filed": r.filed,
                    "accepted": r.accepted, "tickers": "|".join(tickers(text)), "cash": cash_price(text)}
        except Exception as e:                                       # noqa: BLE001 (logged, retried next run)
            print(f"{r.accession}: {e}", flush=True)
            return None
        finally:
            time.sleep(pause)

    rows = list(todo.itertuples())
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(rows), 200):
            got = [x for x in pool.map(one, rows[i:i + 200]) if x]
            done = pd.concat([done, pd.DataFrame(got)], ignore_index=True)
            done.to_parquet(path)
            print(f"{len(done)} filings parsed", flush=True)
    return done


# ---------------------------------------------------------------- deals

def load_prices(tickers_wanted, hf_daily=HF_DAILY) -> dict:
    """Raw daily bars (open, close, volume) of the wanted tickers from the monthly HF files."""
    parts = []
    for p in sorted(Path(hf_daily).glob("*.parquet")):
        d = pd.read_parquet(p, columns=["ticker", "date", "open", "close", "volume"])
        parts.append(d[d["ticker"].isin(tickers_wanted)])
    df = pd.concat(parts, ignore_index=True).dropna(subset=["open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    return {t: g.set_index("date").sort_index()[["open", "close", "volume"]]
            for t, g in df.groupby("ticker")}


@dataclass(frozen=True)
class DealRules:
    max_spread: float = 0.25          # offer / close - 1 must be in (0, max_spread]
    min_price: float = 2.0
    min_dollar_volume: float = 1e6    # median close x volume of the last 20 sessions
    cooldown_days: int = 365          # a ticker's next deal can start this long after the previous one


def find_deals(filings: pd.DataFrame, prices: dict, calendar: pd.DatetimeIndex,
               blank_check=frozenset(), rules: DealRules = DealRules()) -> pd.DataFrame:
    """One row per deal: ticker, cik, form, accession, day (decision day), offer, close, spread.

    Filings are taken in acceptance order; each uses only its own text and prices up to its
    decision day (see module doc)."""
    f = filings[(filings["tickers"].fillna("") != "") & filings["cash"].notna()
                & ~filings["cik"].isin(blank_check)].copy()
    f["day"] = reaction_days(f, calendar)
    f = f.dropna(subset=["day"]).sort_values(["day", "accepted", "accession"])
    dvol = {t: (p["close"] * p["volume"]).rolling(20, min_periods=10).median() for t, p in prices.items()}
    last_start, rows = {}, []
    for r in f.itertuples():
        ok = []
        for t in r.tickers.split("|"):
            p = prices.get(t)
            if p is None or r.day not in p.index:
                continue
            c = float(p.at[r.day, "close"])
            spread = r.cash / c - 1
            if 0 < spread <= rules.max_spread and c >= rules.min_price \
                    and dvol[t].get(r.day, 0) >= rules.min_dollar_volume:
                ok.append((t, c, spread))
        if len(ok) != 1:
            continue
        t, c, spread = ok[0]
        prev = last_start.get(t)
        if prev is not None and (r.day - prev).days < rules.cooldown_days:
            continue
        last_start[t] = r.day
        rows.append({"ticker": t, "cik": r.cik, "form": r.form, "accession": r.accession,
                     "day": r.day, "offer": r.cash, "close": c, "spread": spread})
    return pd.DataFrame(rows, columns=["ticker", "cik", "form", "accession", "day", "offer", "close", "spread"])


def termination_days(deals: pd.DataFrame, eightk: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """First reaction day after each deal's decision day of an 8-K by the deal's company with item 1.02
    (termination of a material agreement) and without 2.01 / 3.01 (completion, delisting); NaT if none."""
    e = eightk[eightk["items"].str.contains("1.02", regex=False)
               & ~eightk["items"].str.contains("2.01|3.01")].copy()
    e["day"] = reaction_days(e, calendar)
    by_cik = {c: np.sort(g["day"].dropna().to_numpy()) for c, g in e.groupby("cik")}
    out = []
    for r in deals.itertuples():
        days = by_cik.get(r.cik, np.array([], dtype="datetime64[ns]"))
        later = days[days > np.datetime64(r.day)]
        out.append(pd.Timestamp(later[0]) if len(later) else pd.NaT)
    return pd.Series(out, index=deals.index, dtype="datetime64[ns]")


# ---------------------------------------------------------------- portfolio

def deal_frames(deals: pd.DataFrame, prices: dict, pre_days: int = 30) -> dict:
    """{deal name: bars} from `pre_days` before the decision day to the end of the price piece that
    contains it (`midcap.segments`: a long gap or a jump means the ticker changed hands)."""
    from .midcap import segments
    out = {}
    for r in deals.itertuples():
        p = prices[r.ticker]
        for s in segments(p):
            if s.index[0] <= r.day <= s.index[-1]:
                out[r.name_] = s.loc[r.day - pd.Timedelta(days=pre_days):]
                break
    return out


def deal_names(deals: pd.DataFrame) -> pd.Series:
    return deals["ticker"] + "@" + pd.to_datetime(deals["day"]).dt.strftime("%Y%m%d")


def schedule(deals: pd.DataFrame, frames: dict, slots: int, max_days: int = 540) -> pd.DataFrame:
    """Deals held: in deal order, a deal is taken if fewer than `slots` taken deals are still open on its
    decision day. Each holds from `day` (weight set at that close) to `exit` (weight 0 at that close): the
    termination day, `day` + `max_days` or the last bar, whichever is first."""
    rows, open_until = [], []
    for r in deals.sort_values(["day", "name_"]).itertuples():
        if r.name_ not in frames:
            continue
        last = frames[r.name_].index[-1]
        if last <= r.day:
            continue
        stop = min(x for x in (r.termination, r.day + pd.Timedelta(days=max_days), last) if pd.notna(x))
        open_until = [x for x in open_until if x > r.day]
        if len(open_until) >= slots:
            continue
        open_until.append(stop)
        rows.append({"name": r.name_, "day": r.day, "exit": stop})
    return pd.DataFrame(rows, columns=["name", "day", "exit"])


@dataclass(frozen=True)
class MergerArb:
    """Weight 1/`slots` in each scheduled deal from its decision day up to (not including) its exit day."""
    held: pd.DataFrame
    slots: int = 20

    def weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        for r in self.held.itertuples():
            if r.name in w.columns:
                w.loc[(w.index >= r.day) & (w.index < r.exit), r.name] = 1.0 / self.slots
        return w


def with_cash_rate(res, closes: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """Daily return of the portfolio plus the risk-free rate on the previous close's cash share."""
    eq = res.equity
    invested = (res.holdings * closes.reindex(columns=res.holdings.columns).ffill()).sum(axis=1)
    cash_share = (1 - invested / eq).clip(lower=0).shift().fillna(1.0)
    r = eq.pct_change().fillna(eq.iloc[0] / res.initial_capital - 1)
    return r + cash_share * rf.reindex(r.index).fillna(0.0)


def deal_returns(res, held: pd.DataFrame) -> pd.DataFrame:
    """Per deal held: entry and exit dates and the return from the buy fills to the sell fills."""
    o = pd.DataFrame(res.orders, columns=["date", "name", "shares", "price", "commission"])
    rows = []
    for name, g in o.groupby("name"):
        buy, sell = g[g["shares"] > 0], g[g["shares"] < 0]
        cost = float((buy["shares"] * buy["price"]).sum() + g["commission"].sum())
        back = float((-sell["shares"] * sell["price"]).sum())
        rows.append({"name": name, "entry": buy["date"].min(), "exit": g["date"].max(),
                     "return": back / cost - 1 if cost else np.nan})
    return held.merge(pd.DataFrame(rows), on="name", how="left")


# ---------------------------------------------------------------- runner

SLOTS = {"merger_arb": 20, "merger_arb_10": 10, "merger_arb_40": 40}    # primary first; no tuning
SPY_FILE = Path("data/history/tiingo/SPY.parquet")


def build_deals(data=DATA, edgar=EDGAR, hf_daily=HF_DAILY, spy=SPY_FILE):
    """filings.parquet -> deals.csv (with termination days) and the deals' price frames."""
    filings = pd.read_parquet(Path(data) / "filings.parquet")
    comp = pd.read_parquet(Path(edgar) / "companies.parquet", columns=["cik", "sic"])
    blank = frozenset(comp.loc[comp["sic"] == BLANK_CHECK, "cik"])
    wanted = {t for s in filings["tickers"].dropna() for t in s.split("|") if t}
    prices = load_prices(wanted, hf_daily)
    calendar = pd.read_parquet(spy).index
    calendar = calendar[(calendar >= START) & (calendar <= max(p.index[-1] for p in prices.values()))]
    deals = find_deals(filings, prices, calendar, blank)
    filed = pd.read_parquet(Path(edgar) / "deal_filings.parquet")
    deals["termination"] = termination_days(deals, filed[filed["form"] == "8-K"], calendar)
    deals["name_"] = deal_names(deals)
    deals.to_csv(Path(data) / "deals.csv", index=False)
    return deals, deal_frames(deals, prices), calendar


def run(slots: int, deals, frames, calendar, rf, spy=SPY_FILE) -> dict:
    held = schedule(deals, frames, slots)
    cal = pd.read_parquet(spy)[["open", "close"]].reindex(calendar).dropna()
    fr = {n: frames[n] for n in held["name"]} | {"SPY": cal}
    closes = pd.DataFrame({t: f["close"] for t, f in fr.items()}).sort_index()
    out = {"held": held}
    for var, cost in COSTS.items():
        res = PortfolioBacktester(MergerArb(held, slots), cost, CAPITAL).run(fr, start=START)
        out[var] = with_cash_rate(res, closes, rf)
        if var == "base":
            out["deals"] = deal_returns(res, held)
            out["turnover"] = res.turnover
    out["spy"] = cal["close"].pct_change().reindex(out["base"].index)
    return out


def summarize(name: str, res: dict, factors: pd.DataFrame) -> dict:
    from .french import factor_regression
    from .stats import returns_stats, sharpe_interval
    rf = factors["RF"]
    f = factors.assign(MktRF=factors["Mkt"] - factors["RF"])
    cols = ("MktRF", "SMB", "HML", "RMW", "CMA", "Mom")
    row = {"experiment": name}
    for var in ("base", "gross", "stress"):
        r = res[var]
        ex = (r - rf.reindex(r.index)).dropna()
        st = returns_stats(r)
        row |= {f"{var}_cagr": st["cagr"], f"{var}_excess_yr": ex.mean() * 252,
                f"{var}_sharpe_excess": ex.mean() / ex.std() * np.sqrt(252)}
        if var == "base":
            lo, hi = sharpe_interval(ex)
            fr = factor_regression(ex, f, cols)
            row |= {"max_drawdown": st["max_drawdown"], "sharpe_lo": lo, "sharpe_hi": hi,
                    "ff_alpha": fr["alpha"], "ff_alpha_t": fr["alpha_t"], "ff_beta": fr["MktRF"]}
    d = res["deals"]
    row |= {"deals": len(d), "deal_mean": d["return"].mean(), "deal_median": d["return"].median(),
            "deals_lost_10pct": float((d["return"] < -0.10).mean()), "turnover": res["turnover"]}
    return row


def fingerprint() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    for mod in ("portfolio", "costs", "pead", "midcap", "french", "stats"):
        h.update(Path(sys.modules[f"algotrader.{mod}"].__file__).read_bytes())
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("fetch", "deals", "run"))
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    if a.command == "fetch":
        fetch()
        return
    deals, frames, calendar = build_deals()
    print(f"{len(deals)} deals, {len(frames)} with prices", flush=True)
    if a.command == "deals":
        return
    from . import french, stats  # noqa: F401 (fingerprint)
    factors = french.load()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, man = [], {}
    for name, slots in SLOTS.items():
        res = run(slots, deals, frames, calendar, factors["RF"])
        pd.DataFrame({k: res[k] for k in ("base", "gross", "stress", "spy")}).to_csv(
            out / f"{name}_returns.csv", float_format="%.6g")
        res["deals"].to_csv(out / f"{name}_deals.csv", index=False, float_format="%.6g")
        rows.append(summarize(name, res, factors))
        man[name] = {"configurations": 1, "code_sha256": fingerprint(), "slots": slots}
        print(rows[-1], flush=True)
    deals.drop(columns="name_").to_csv(out / "deals.csv", index=False, float_format="%.6g")
    (out / "data.json").write_text(json.dumps(
        {"filings_sha256": hashlib.sha256((DATA / "filings.parquet").read_bytes()).hexdigest()}, indent=2))
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
