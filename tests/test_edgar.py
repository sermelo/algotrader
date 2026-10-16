import io
import json
import zipfile

import pandas as pd
import pytest

from algotrader import edgar

CHANGES_HTML = """<table><tr><th>Symbol</th><th>Security</th></tr><tr><td>AAA</td><td>A</td></tr></table>
<table>
<tr><th rowspan=2>Date</th><th colspan=2>Added</th><th colspan=2>Removed</th><th rowspan=2>Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>March 3, 2015</td><td>NEW</td><td>New Thing Inc.</td><td>OLD</td><td>Old Widgets Corporation</td><td>merger</td></tr>
<tr><td>June 1, 2012</td><td>ZZZ</td><td>Zed Co</td><td></td><td></td><td>added</td></tr>
<tr><td>May 1, 2020</td><td></td><td></td><td>BBF</td><td>Big Bank</td><td>acquired</td></tr>
</table>"""


def test_normalize_name():
    assert edgar.normalize_name("The Procter & Gamble Company") == "procter gamble"
    assert edgar.normalize_name("CABLEVISION SYSTEMS CORP /NY") == "cablevision systems"
    assert edgar.normalize_name("Marsh & McLennan Companies") == edgar.normalize_name("Marsh and McLennan Companies")
    assert edgar.normalize_name("Old Widgets Corp.") == edgar.normalize_name("OLD WIDGETS CORPORATION")
    assert edgar.normalize_name("Alphabet Inc. (Class A)[3]") == "alphabet"


def test_parse_changes():
    c = edgar.parse_changes(CHANGES_HTML)
    assert set(c["ticker"]) == {"NEW", "OLD", "ZZZ", "BBF", "AAA"}
    assert c[c["ticker"] == "AAA"].iloc[0]["action"] == "member"
    old = c[c["ticker"] == "OLD"].iloc[0]
    assert old["action"] == "rem" and old["name"] == "Old Widgets Corporation"
    assert old["date"] == pd.Timestamp("2015-03-03")


def _sub_zip(path):
    files = {
        "CIK0000000001.json": {"cik": "1", "name": "Old Widgets Corp", "tickers": [],
                               "formerNames": [], "sic": "3500",
                               "filings": {"recent": {"form": ["10-K", "8-K"],
                                                      "filingDate": ["2013-02-01", "2013-01-25"],
                                                      "items": ["", "2.02,9.01"],
                                                      "acceptanceDateTime": ["", "2013-01-25T16:05:00.000Z"]}}},
        "CIK0000000001-submissions-001.json": {"form": ["10-Q"], "filingDate": ["2009-05-01"],
                                               "items": [""], "acceptanceDateTime": [""]},
        "CIK0000000002.json": {"cik": "2", "name": "Canadian Banc Fund", "tickers": ["BK"],
                               "formerNames": [], "filings": {"recent": {
                                   "form": ["10-K"], "filingDate": ["2026-03-01"]}}},
        "CIK0000000003.json": {"cik": "3", "name": "Big Bank Inc", "tickers": ["BIGB"],
                               "formerNames": [{"name": "Bank of Things"}], "filings": {"recent": {
                                   "form": ["10-Q", "10-K"], "filingDate": ["2008-05-01", "2025-02-01"]}}},
    }
    with zipfile.ZipFile(path, "w") as z:
        for n, d in files.items():
            z.writestr(n, json.dumps(d))


def test_index_submissions_reads_history_and_earnings(tmp_path):
    _sub_zip(tmp_path / "s.zip")
    comp, earn = edgar.index_submissions(tmp_path / "s.zip")
    c1 = comp.set_index("cik").loc[1]
    assert c1["first_report"] == "2009-05-01" and c1["last_report"] == "2013-02-01"   # older file used
    assert comp.set_index("cik").loc[3, "names"] == "bank of things|big bank"
    assert earn.to_dict("records") == [{"cik": 1, "filed": "2013-01-25",
                                        "accepted": "2013-01-25T16:05:00.000Z", "form": "8-K"}]


def test_map_members_rejects_reused_ticker_and_matches_names(tmp_path):
    _sub_zip(tmp_path / "s.zip")
    comp, _ = edgar.index_submissions(tmp_path / "s.zip")
    spans = pd.DataFrame({"first": pd.to_datetime(["2008-01-02", "2008-01-02", "2010-01-04"]),
                          "last": pd.to_datetime(["2015-03-02", "2024-12-31", "2012-01-03"])},
                         index=["OLD", "BIGB", "BK"])
    spans.loc["BBF"] = [pd.Timestamp("2009-01-02"), pd.Timestamp("2020-04-30")]
    spans.loc["CUR"] = [pd.Timestamp("2009-01-02"), pd.Timestamp("2020-04-30")]
    changes = edgar.parse_changes(CHANGES_HTML)
    m = edgar.map_members(spans, comp, changes, current={"CUR": 3})
    assert m.loc["BBF", "cik"] == 3 and m.loc["BBF", "method"] == "name"   # prefix of "big bank inc"
    assert m.loc["CUR", "cik"] == 3 and m.loc["CUR", "method"] == "ticker"  # company_tickers.json
    assert m.loc["OLD", "cik"] == 1 and m.loc["OLD", "method"] == "name"
    assert m.loc["BIGB", "cik"] == 3 and m.loc["BIGB", "method"] == "ticker"
    assert m.loc["BK", "cik"] is None or pd.isna(m.loc["BK", "cik"])   # BK's filer only from 2026


def test_extract_facts(tmp_path):
    d = {"cik": 1, "facts": {
        "us-gaap": {"StockholdersEquity": {"units": {"USD": [
            {"end": "2012-12-31", "val": 500, "filed": "2013-02-01", "form": "10-K", "fy": 2012,
             "fp": "FY", "accn": "x1"}]}},
            "Goodwill": {"units": {"USD": [{"end": "2012-12-31", "val": 1, "filed": "2013-02-01"}]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2013-01-20", "val": 1000, "filed": "2013-02-01", "form": "10-K"}]}}}}}
    with zipfile.ZipFile(tmp_path / "f.zip", "w") as z:
        z.writestr("CIK0000000001.json", json.dumps(d))
        z.writestr("CIK0000000009.json", json.dumps(d | {"cik": 9}))
    df = edgar.extract_facts(tmp_path / "f.zip", [1])
    assert set(df["tag"]) == {"StockholdersEquity", "EntityCommonStockSharesOutstanding"}
    assert set(df["cik"]) == {1}                                   # other companies skipped
    eq = df[df["tag"] == "StockholdersEquity"].iloc[0]
    assert eq["value"] == 500 and eq["filed"] == pd.Timestamp("2013-02-01")


def test_download_sends_user_agent(tmp_path):
    (tmp_path / "ua").write_text("Someone someone@example.com\n")
    seen = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req):
        seen.append((req.full_url, req.get_header("User-agent")))
        return Resp(b"data")

    saved = edgar.download(tmp_path / "out", ua_file=tmp_path / "ua", opener=opener)
    assert len(saved) == 4 and (tmp_path / "out" / "companyfacts.zip").read_bytes() == b"data"
    assert all(ua == "Someone someone@example.com" for u, ua in seen if "sec.gov" in u)
