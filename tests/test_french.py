import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from algotrader import french

FF5_TEXT = """This file was created by using the 202608 CRSP database.
Some description.

,Mkt-RF,SMB,HML,RMW,CMA,RF
20240102,   1.00,    0.50,   -0.20,   0.10,    0.00,    0.02
20240103,  -2.00,    0.00,    0.30,  -0.10,    0.20,    0.02

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
2024,   10.0,    1.0,   1.0,   1.0,    1.0,    5.0

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

MOM_TEXT = """Momentum description
with several lines.

,Mom
20240102,   0.40
20240103,  -0.60
"""


def test_parse_reads_the_daily_table_in_decimals():
    df = french.parse(FF5_TEXT)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df.loc["2024-01-02", "Mkt-RF"] == pytest.approx(0.01)
    assert df.loc["2024-01-03", "CMA"] == pytest.approx(0.002)      # annual rows not read


def _zip(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("file.csv", text)
    return buf.getvalue()


def test_load_joins_files_and_uses_total_market_return(tmp_path):
    (tmp_path / french.FILES[0]).write_bytes(_zip(FF5_TEXT))
    (tmp_path / french.FILES[1]).write_bytes(_zip(MOM_TEXT))
    df = french.load(tmp_path)
    assert list(df.columns) == list(french.FACTORS) + ["RF"]
    assert df.loc["2024-01-02", "Mkt"] == pytest.approx(0.0102)    # 1.00% + 0.02%
    assert df.loc["2024-01-03", "Mom"] == pytest.approx(-0.006)


def test_download_saves_zip_files(tmp_path):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = []

    def opener(req):
        seen.append(req.full_url)
        return Resp(_zip(FF5_TEXT))

    paths = french.download(tmp_path, opener=opener)
    assert [p.name for p in paths] == list(french.FILES)
    assert all(u.startswith(french.BASE) for u in seen)
    with pytest.raises(RuntimeError):
        french.download(tmp_path, opener=lambda req: Resp(b"<html>"))


def _factors(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, len(french.FACTORS))), index=idx,
                        columns=french.FACTORS)


def test_regression_recovers_known_alpha_and_loadings():
    f = _factors()
    r = 0.0004 + 0.8 * f["Mkt"] - 0.3 * f["HML"] + 0.5 * f["Mom"]
    out = french.factor_regression(r, f)
    assert out["alpha"] == pytest.approx(0.0004 * 252)
    assert out["Mkt"] == pytest.approx(0.8)
    assert out["HML"] == pytest.approx(-0.3)
    assert out["Mom"] == pytest.approx(0.5)
    assert out["SMB"] == pytest.approx(0.0, abs=1e-9)
    assert out["r2"] == pytest.approx(1.0)
    assert out["days"] == 500


def test_regression_uses_common_days_and_noise_gives_small_t():
    f = _factors()
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 300), index=f.index[100:400])
    out = french.factor_regression(r, f)
    assert out["days"] == 300
    assert abs(out["alpha_t"]) < 3


def test_regression_without_data_is_nan():
    f = _factors(10)
    out = french.factor_regression(pd.Series(0.0, index=f.index), f)
    assert np.isnan(out["alpha"])


def test_report_reads_walk_forward_returns(tmp_path):
    f = _factors()
    rng = np.random.default_rng(2)
    hold = f["Mkt"] + rng.normal(0, 0.001, len(f))
    wf = pd.DataFrame({"base": 0.5 * f["Mkt"], "gross": 0.5 * f["Mkt"], "stress": 0.5 * f["Mkt"],
                       "hold": hold})
    wf.to_csv(tmp_path / "toy_wf_returns.csv")
    table = french.regress_results(tmp_path, f)
    assert list(table.index) == ["toy", "buy_and_hold"]
    assert table.loc["toy", "Mkt"] == pytest.approx(0.5)
    assert table.loc["toy", "alpha"] == pytest.approx(0.0, abs=1e-12)
    path = french.write_report(tmp_path, f, data_dir=tmp_path)
    text = path.read_text()
    assert "`toy`" in text and "`buy_and_hold`" in text
    assert (tmp_path / "factor_alpha.csv").exists()
