import io
import urllib.error
import zipfile

import numpy as np
import pandas as pd
import pytest

from algotrader import history

MEMBERS = 'date,tickers\n2007-12-31,"AAA,BBB"\n2008-01-02,"AAA,BBB,BF.B"\n2008-06-02,"AAA,CCC"\n'


def test_members_and_spans():
    m = history.parse_members(MEMBERS)
    assert len(m) == 7
    spans = history.member_spans(m, start="2008-01-01")
    assert list(spans.index) == ["AAA", "BBB", "BF.B", "CCC"]
    assert spans.loc["BBB", "last"] == pd.Timestamp("2008-01-02")
    assert spans.loc["AAA", "first"] == pd.Timestamp("2008-01-02")   # 2007 row ignored
    assert history.tiingo_symbol("BF.B") == "BF-B"
    assert history.wiki_symbol("BF.B") == "BF_B"


def _wiki_zip(path):
    header = "ticker,date,open,high,low,close,volume,ex-dividend,split_ratio,adj_open,adj_high,adj_low,adj_close,adj_volume\n"
    rows = []
    for t in ("AAA", "BBB", "CCC"):
        for i, d in enumerate(["2010-01-04", "2010-01-05", "2010-01-06"]):
            p = 10 + i
            rows.append(f"{t},{d},{p},{p},{p},{p},100,0,1,{p/2},{p/2},{p/2},{p/2},200")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("WIKI_PRICES.csv", header + "\n".join(rows) + "\n")


def test_wiki_extract_streams_by_ticker(tmp_path):
    _wiki_zip(tmp_path / "w.zip")
    written = history.wiki_extract(tmp_path / "w.zip", tmp_path / "out", tickers=["AAA", "CCC"],
                                   chunksize=2)      # chunks split tickers
    assert written == ["AAA", "CCC"]
    df = pd.read_parquet(tmp_path / "out" / "CCC.parquet")
    assert list(df.columns) == history.COLUMNS
    assert len(df) == 3
    assert df["close"].iloc[-1] == 6.0                # adjusted column
    assert df["volume"].iloc[0] == 200
    assert not (tmp_path / "out" / "BBB.parquet").exists()


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_limiter_spaces_requests_and_caps_symbols(tmp_path):
    c = Clock()
    lim = history.TiingoLimiter(tmp_path / "log.csv", per_month=2, per_day=3, interval=75,
                                clock=c, sleep=c.sleep)
    for s in ("A", "B"):
        assert lim.allow(s)
        lim.wait()
        lim.record(s)
    assert c.slept == [75]
    assert not lim.allow("C")              # third new symbol this month
    assert lim.allow("A")                  # repeat symbol is fine
    lim.record("A")
    assert not lim.allow("A")              # 3 requests today
    again = history.TiingoLimiter(tmp_path / "log.csv", per_month=2, clock=c, sleep=c.sleep)
    assert len(again.log) == 3 and not again.allow("C")   # log persists


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


TIINGO_CSV = (b"date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor\n"
              b"2013-11-07,44.9,50.09,44.0,45.1,100,22.45,25.0,22.0,22.5,200,0.0,1.0\n"
              b"2013-11-08,41.65,46.94,40.685,45.93,100,20.8,23.4,20.3,22.9,200,0.0,1.0\n")


def test_tiingo_download_saves_logs_and_stops(tmp_path):
    (tmp_path / "token").write_text("secret\n")
    c = Clock()
    lim = history.TiingoLimiter(tmp_path / "log.csv", per_month=10, interval=0, clock=c, sleep=c.sleep)
    seen = []

    def opener(req):
        seen.append(req)
        if "/GONE/" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
        return Resp(TIINGO_CSV)

    st = history.tiingo_download(["AAA", "GONE", "BBB"], tmp_path / "t", lim,
                                 token_file=tmp_path / "token", opener=opener, max_new=2)
    assert st == {"AAA": "ok", "GONE": "error 404"}
    assert seen[0].get_header("Authorization") == "Token secret"
    df = pd.read_parquet(tmp_path / "t" / "AAA.parquet")
    assert df["close"].tolist() == [22.45, 20.8] and list(df.columns) == history.COLUMNS
    assert (tmp_path / "t" / "GONE.missing").exists()
    st = history.tiingo_download(["AAA", "GONE", "BBB"], tmp_path / "t", lim,
                                 token_file=tmp_path / "token", opener=opener)
    assert st == {"BBB": "ok"}                       # resumes, skips done and missing


def _series(r, start="2010-01-04"):
    idx = pd.bdate_range(start, periods=len(r))
    return pd.DataFrame({"close": 100 * np.cumprod(1 + np.asarray(r))}, index=idx)


def test_overlap_check_tells_same_from_different_company():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 300)
    a, b = _series(r), _series(r * 1.0)
    b["close"] *= 0.5                                  # different adjustment level, same returns
    same = history.overlap_check(a, b)
    assert same["days"] == 299 and same["corr"] == pytest.approx(1.0)
    other = history.overlap_check(a, _series(rng.normal(0, 0.02, 300)))
    assert abs(other["corr"]) < 0.3


def test_check_flags_mismatch_and_partial(tmp_path):
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, 600)
    out = tmp_path / "h"
    (out / "wiki").mkdir(parents=True)
    (out / "tiingo").mkdir()
    _series(r).to_parquet(out / "tiingo" / "SPY.parquet")
    _series(r[:400]).to_parquet(out / "wiki" / "AAA.parquet")
    _series(r).to_parquet(out / "tiingo" / "AAA.parquet")                 # agrees: ok
    _series(r[:400]).to_parquet(out / "wiki" / "BBB.parquet")
    _series(rng.normal(0, 0.02, 600)).to_parquet(out / "tiingo" / "BBB.parquet")  # reused ticker
    _series(r[:100]).to_parquet(out / "tiingo" / "CCC.parquet")           # too short
    idx = pd.bdate_range("2010-01-04", periods=600)
    spans = pd.DataFrame({"first": [idx[0]] * 4, "last": [idx[-1]] * 4},
                         index=["AAA", "BBB", "CCC", "DDD"])
    t = history.check(spans, out, hf_dir=tmp_path / "none")
    assert t["status"].to_dict() == {"AAA": "ok", "BBB": "mismatch", "CCC": "partial",
                                     "DDD": "missing"}
