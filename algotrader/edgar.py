"""Point-in-time fundamentals and earnings dates from SEC EDGAR (roadmap item D3).

Free bulk files from sec.gov (rebuilt nightly): `companyfacts.zip` (every XBRL fact
each company reported, with the date it was filed) and `submissions.zip` (each
company's names, tickers and filing history, including 8-K item 2.02, "Results of
Operations", i.e. earnings releases). SEC requires a User-Agent with a contact email,
read from ~/.config/sec/user_agent (outside the repo).

Tickers are reused and dropped by EDGAR after delisting, so S&P 500 members are mapped
to a company id (CIK) by their ticker when EDGAR still lists it, else by the company
name of the S&P 500 change list (Wikipedia), and a mapping is accepted only if the
company filed 10-K or 10-Q reports during the ticker's membership.

Example (on the VPS; the files are 3 GB):
    python -m algotrader.edgar download    # -> data/edgar/
    python -m algotrader.edgar index       # -> data/edgar/companies.parquet, earnings.parquet
    python -m algotrader.edgar map         # -> data/edgar/cik_map.csv
    python -m algotrader.edgar facts       # -> data/edgar/facts.parquet
"""
import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FILES = {
    "companyfacts.zip": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "submissions.zip": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
    "company_tickers.json": "https://www.sec.gov/files/company_tickers.json",
}
# A fixed revision of Wikipedia's "List of S&P 500 companies" that still has the table
# of changes (ticker and company name of every addition and removal).
WIKI_REVISION = 1308482549
WIKI_URL = f"https://en.wikipedia.org/w/index.php?oldid={WIKI_REVISION}"
UA_FILE = Path.home() / ".config" / "sec" / "user_agent"
OUT = Path("data/edgar")
REPORTS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT", "20-F", "40-F")
TAGS = {
    "dei": ("EntityCommonStockSharesOutstanding",),
    "us-gaap": ("StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                "Assets", "Liabilities", "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss",
                "EarningsPerShareBasic", "EarningsPerShareDiluted",
                "CommonStockSharesOutstanding",
                "WeightedAverageNumberOfDilutedSharesOutstanding"),
}


def user_agent(path=UA_FILE) -> str:
    return Path(path).read_text().strip()


def download(out_dir=OUT, ua_file=UA_FILE, opener=urllib.request.urlopen) -> list:
    """Fetch the SEC bulk files and the Wikipedia change list into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ua = user_agent(ua_file)
    saved = []
    for name, url in FILES.items():
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with opener(req) as resp, open(out / (name + ".part"), "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        (out / (name + ".part")).replace(out / name)
        saved.append(out / name)
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "algotrader/0.1"})
    with opener(req) as resp:
        (out / "sp500_changes.html").write_bytes(resp.read())
    saved.append(out / "sp500_changes.html")
    return saved


def parse_changes(html: str) -> pd.DataFrame:
    """Ticker/company-name pairs from the Wikipedia page: every addition and removal of
    the change table (with its date) and the current members table (action 'member')."""
    tables = pd.read_html(io.StringIO(html))
    t = next(x for x in tables if any("Removed" in str(c) for c in x.columns))
    t.columns = ["date", "add_ticker", "add_name", "rem_ticker", "rem_name", "reason"][:t.shape[1]]
    t["date"] = pd.to_datetime(t["date"], format="mixed", errors="coerce")
    rows = []
    for _, r in t.iterrows():
        for action in ("add", "rem"):
            tick, name = r[f"{action}_ticker"], r[f"{action}_name"]
            if isinstance(tick, str) and tick.strip() and isinstance(name, str):
                rows.append((r["date"], tick.strip(), name.strip(), action))
    cur = next((x for x in tables if "Symbol" in x.columns and "Security" in x.columns), None)
    if cur is not None:
        for _, r in cur.iterrows():
            rows.append((pd.NaT, str(r["Symbol"]).strip(), str(r["Security"]).strip(), "member"))
    return pd.DataFrame(rows, columns=["date", "ticker", "name", "action"])


_SUFFIX = re.compile(r"\b(the|incorporated|inc|corporation|corp|company|co|ltd|limited|plc|"
                     r"holdings?|group|lp|llc|sa|nv|ag|se|de|del|new)\b")


def normalize_name(name: str) -> str:
    """Lower case; '&'/'and', punctuation, state tags such as '/NY/' and legal suffixes
    removed; single spaces."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"\[.*?\]|\(.*?\)|/[a-z]{2,3}/?$", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return " ".join(w for w in s.split() if w != "and")


# ---------------------------------------------------------------- submissions

def _filing_arrays(d: dict) -> dict:
    return d["filings"]["recent"] if "filings" in d else d


def index_submissions(zip_path):
    """Per company: names (current and former, normalized), tickers, first and last
    periodic report; and the earnings releases (8-K with item 2.02).

    Each file is summarised as it is read (the archive has about a million).
    Returns (companies, earnings) DataFrames.
    """
    companies, spans, earnings = {}, {}, []

    def scan(cik, a):
        forms, dates = a.get("form", []), a.get("filingDate", [])
        items = a.get("items") or [""] * len(forms)
        accept = a.get("acceptanceDateTime") or [""] * len(forms)
        first, last = spans.get(cik, (None, None))
        for form, date, item, acc in zip(forms, dates, items, accept):
            if form in REPORTS:
                first = date if first is None or date < first else first
                last = date if last is None or date > last else last
            if form in ("8-K", "8-K/A") and "2.02" in (item or ""):
                earnings.append((cik, date, acc, form))
        spans[cik] = (first, last)

    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.startswith("CIK"):
                continue
            d = json.loads(z.read(n))
            cik = int(n[3:13])
            scan(cik, _filing_arrays(d))
            if "-submissions-" in n:
                continue
            names = [d.get("name") or ""] + [f.get("name") or "" for f in d.get("formerNames") or []]
            companies[cik] = {"cik": cik, "name": d.get("name"),
                              "names": "|".join(sorted({normalize_name(x) for x in names if x})),
                              "tickers": "|".join(t.upper() for t in d.get("tickers") or []),
                              "sic": d.get("sic")}
    for cik, c in companies.items():
        c["first_report"], c["last_report"] = spans.get(cik, (None, None))
    comp = pd.DataFrame(list(companies.values()))
    earn = pd.DataFrame(earnings, columns=["cik", "filed", "accepted", "form"]).drop_duplicates()
    return comp, earn


# ---------------------------------------------------------------- mapping

def map_members(spans: pd.DataFrame, companies: pd.DataFrame, changes: pd.DataFrame,
                current: dict | None = None) -> pd.DataFrame:
    """CIK for each member ticker: by EDGAR ticker (submissions, or `current`, SEC's
    company_tickers.json as {ticker: cik}), else by the Wikipedia company name, exact
    after normalisation or else as a whole-word prefix of one another. Only companies with
    periodic reports overlapping the membership are accepted, and a ticker or name must
    match exactly one such company."""
    comp = companies.dropna(subset=["first_report"]).copy()
    comp["first_report"] = pd.to_datetime(comp["first_report"])
    comp["last_report"] = pd.to_datetime(comp["last_report"])
    by_ticker, by_name = {}, {}
    for _, c in comp.iterrows():
        for t in filter(None, c["tickers"].split("|")):
            by_ticker.setdefault(t, []).append(c)
        for nm in {normalize_name(x) for x in c["names"].split("|") if x}:   # idempotent
            by_name.setdefault(nm, []).append(c)
    for t, cik in (current or {}).items():
        c = comp[comp["cik"] == cik]
        if len(c):
            by_ticker.setdefault(t.upper(), []).append(c.iloc[0])
    names = changes.groupby("ticker")["name"].agg(lambda s: list(dict.fromkeys(s)))
    all_names = sorted(by_name)
    by_joined = {}
    for nm, cs in by_name.items():
        by_joined.setdefault(nm.replace(" ", ""), []).extend(cs)

    def name_candidates(nm):
        key = normalize_name(nm)
        exact = by_name.get(key, []) + by_joined.get(key.replace(" ", ""), [])
        if exact or not key:
            return exact
        return [c for other in all_names
                if other.startswith(key + " ") or key.startswith(other + " ")
                for c in by_name[other]]

    def overlaps(c, first, last):
        return c["first_report"] <= last + pd.Timedelta(days=120) and \
            c["last_report"] >= first - pd.Timedelta(days=120)

    rows = []
    for ticker, (first, last) in spans.iterrows():
        rec = {"ticker": ticker, "cik": None, "method": None, "company": None}
        for t in (ticker, ticker.replace(".", "-")):
            cands = {int(c["cik"]): c for c in by_ticker.get(t, []) if overlaps(c, first, last)}
            if len(cands) == 1:
                c = next(iter(cands.values()))
                rec |= {"cik": int(c["cik"]), "method": "ticker", "company": c["name"]}
                break
        if rec["cik"] is None:
            for nm in names.get(ticker, []):
                cands = {int(c["cik"]): c for c in name_candidates(nm) if overlaps(c, first, last)}
                if len(cands) == 1:
                    c = next(iter(cands.values()))
                    rec |= {"cik": int(c["cik"]), "method": "name", "company": c["name"]}
                    break
        rows.append(rec)
    return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------- facts

def extract_facts(zip_path, ciks, tags=TAGS) -> pd.DataFrame:
    """Long table of the selected XBRL facts for `ciks`: cik, taxonomy, tag, unit,
    start, end, value, filed, form, fy, fp, accession. `filed` is when the value became
    public: use only facts filed on or before a decision date."""
    want = {f"CIK{int(c):010d}.json" for c in ciks}
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n not in want:
                continue
            d = json.loads(z.read(n))
            cik = int(d.get("cik", n[3:13]))
            for tax, names in tags.items():
                facts = d.get("facts", {}).get(tax, {})
                for tag in names:
                    for unit, vals in facts.get(tag, {}).get("units", {}).items():
                        for v in vals:
                            rows.append((cik, tax, tag, unit, v.get("start"), v.get("end"),
                                         v.get("val"), v.get("filed"), v.get("form"),
                                         v.get("fy"), v.get("fp"), v.get("accn")))
    df = pd.DataFrame(rows, columns=["cik", "taxonomy", "tag", "unit", "start", "end", "value",
                                     "filed", "form", "fy", "fp", "accession"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


# ---------------------------------------------------------------- CLI

def main(argv=None):
    from .history import member_spans, parse_members

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["download", "index", "map", "facts"])
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--members", default="data/history/sp500_members.csv")
    p.add_argument("--start", default="2008-01-01")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.action == "download":
        for f in download(out):
            print("saved", f)
    elif args.action == "index":
        comp, earn = index_submissions(out / "submissions.zip")
        comp.to_parquet(out / "companies.parquet")
        earn.to_parquet(out / "earnings.parquet")
        print(f"{len(comp)} companies, {len(earn)} earnings releases")
    elif args.action == "map":
        spans = member_spans(parse_members(Path(args.members).read_text()), args.start)
        changes = parse_changes((out / "sp500_changes.html").read_text())
        cur = {v["ticker"]: int(v["cik_str"]) for v in
               json.loads((out / "company_tickers.json").read_text()).values()}
        m = map_members(spans, pd.read_parquet(out / "companies.parquet"), changes, cur)
        m.to_csv(out / "cik_map.csv")
        print(m["method"].fillna("unmapped").value_counts().to_string())
    else:
        m = pd.read_csv(out / "cik_map.csv").dropna(subset=["cik"])
        df = extract_facts(out / "companyfacts.zip", m["cik"].astype(int).unique())
        df.to_parquet(out / "facts.parquet")
        print(f"{len(df)} facts for {df['cik'].nunique()} companies")


if __name__ == "__main__":
    main()
