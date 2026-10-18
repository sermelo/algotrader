import json

from algotrader import trials


def test_counts_all_sources_and_reports_bar(tmp_path):
    core = {"sma_daily": {}, "rsi2_daily": {}}
    (tmp_path / "core.json").write_text(json.dumps(core))
    (tmp_path / "long.json").write_text(json.dumps({"x": {"configurations": 7}}))
    src = {"core": tmp_path / "core.json", "broad": tmp_path / "missing.json",
           "long": tmp_path / "long.json", "long_stop": tmp_path / "stop.json"}
    (tmp_path / "stop.json").write_text(json.dumps({"y": {"configurations": 3}}))
    c = trials.count(src)
    from algotrader.experiments import BY_NAME
    n_core = len(BY_NAME["sma_daily"].combos()) + len(BY_NAME["rsi2_daily"].combos())
    assert c == {"core": (2, n_core), "long": (1, 7), "long_stop": (1, 3)}
    text = trials.report(src, long_years=10.0)
    assert f"**{n_core + 10}**" in text and "10.0 years" in text and "3.3 years" in text
