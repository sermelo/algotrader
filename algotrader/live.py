"""Live signal for paper trading: today's target portfolio of a fundamental sort.

The backtest's `FundamentalSort` on the long history, computed for one decision day from current
data instead of the archived bulk files:

- **members**: today's S&P 500 list (Wikipedia's "List of S&P 500 companies" table, which gives
  each member's SEC company id), instead of the dated membership lists, which end in Aug 2025;
- **facts**: each member's SEC company-facts JSON (data.sec.gov API, one request per company at
  most `SEC_RATE` a second, cached for the day), read by the same parser as the bulk file
  (`edgar.extract_facts`) and turned into the signal by the same code (`fundamentals.signals`), so a
  value counts only from the day after it was filed;
- **choice**: the `top_n` highest (lowest if not `largest`) scores, ties in member order, as
  `_top_weights` ranks them.

    python -m algotrader.live --top-n 10        # print today's target and write data/paper/target_<day>.json
"""
import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .edgar import UA_FILE, extract_facts, user_agent
from .fundamentals import signals

MEMBERS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_RATE = 8                      # requests a second (SEC allows 10)
OUT = Path("data/paper")


def parse_members(html: str) -> pd.DataFrame:
    """Ticker -> CIK of the current S&P 500 members (index: ticker as listed, e.g. 'BRK.B')."""
    t = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    t = t.rename(columns={"Symbol": "ticker", "CIK": "cik", "Security": "name", "GICS Sector": "sector"})
    t["ticker"] = t["ticker"].astype(str).str.strip()
    t["cik"] = pd.to_numeric(t["cik"], errors="coerce").astype("Int64")
    if len(t) < 480 or t["cik"].isna().any():
        raise ValueError(f"unexpected member table: {len(t)} rows, {int(t['cik'].isna().sum())} without CIK")
    return t.set_index("ticker")[["cik", "name", "sector"]]


def _get(url: str, ua: str, opener=urllib.request.urlopen) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    with opener(req, timeout=60) as r:
        return r.read()


def fetch_members(opener=urllib.request.urlopen) -> pd.DataFrame:
    return parse_members(_get(MEMBERS_URL, "algotrader/0.1 (research)", opener).decode("utf-8"))


def fetch_facts_zip(ciks, path: Path, ua_file=UA_FILE, opener=urllib.request.urlopen,
                    sleep=time.sleep) -> Path:
    """Company-facts JSON of each CIK into one zip in the bulk file's layout (CIK##########.json),
    so `edgar.extract_facts` reads it. Kept if `path` already exists (one download a day)."""
    path = Path(path)
    if path.exists():
        return path
    ua = user_agent(ua_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for cik in sorted({int(c) for c in ciks}):
            z.writestr(f"CIK{cik:010d}.json", _get(FACTS_URL.format(cik=cik), ua, opener))
            sleep(1 / SEC_RATE)
    tmp.rename(path)
    return path


def scores(facts: pd.DataFrame, members: pd.DataFrame, day, signal="operating_profitability") -> pd.Series:
    """The signal of every member on `day`, from facts filed before `day` (in member order)."""
    day = pd.Timestamp(day).normalize()
    tickers = list(members.index)
    raw_close = pd.DataFrame(1.0, index=pd.DatetimeIndex([day]), columns=tickers)   # dates and columns only
    panels = signals(facts, members[["cik"]].astype("float"), raw_close)
    return panels[signal].iloc[0].reindex(tickers)


def choose(score: pd.Series, top_n: int, largest=True) -> list:
    """The top_n tickers, ties in the series' order (`rank(method='first')`, as `_top_weights`)."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    r = score.dropna().rank(ascending=not largest, method="first")
    return list(r[r <= top_n].sort_values().index)


def target(day, top_n=10, signal="operating_profitability", largest=True, out=OUT,
           members=None, facts=None) -> dict:
    """Today's target: {'day', 'signal', 'top_n', 'tickers', 'scores', 'members'}; written to
    `out/target_<day>.json`."""
    day = pd.Timestamp(day).normalize()
    out = Path(out)
    members = fetch_members() if members is None else members
    if facts is None:
        z = fetch_facts_zip(members["cik"], out / "facts" / f"companyfacts_{day.date()}.zip")
        facts = extract_facts(z, members["cik"].astype(int))
    s = scores(facts, members, day, signal)
    picks = choose(s, top_n, largest)
    res = {"day": str(day.date()), "signal": signal, "top_n": top_n, "largest": largest,
           "tickers": picks, "scores": {t: float(s[t]) for t in picks},
           "names": {t: str(members.loc[t, "name"]) for t in picks},
           "members": len(members), "scored": int(s.notna().sum())}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"target_{day.date()}.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--day", default=str(pd.Timestamp.now(tz="America/New_York").date()))
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--signal", default="operating_profitability")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    res = target(a.day, a.top_n, a.signal, out=a.out)
    print(f"{res['day']}: {res['scored']} of {res['members']} members scored; top {res['top_n']} by {res['signal']}:")
    for t in res["tickers"]:
        print(f"  {t:6s} {res['scores'][t]:8.3f}  {res['names'][t]}")


if __name__ == "__main__":
    main()
