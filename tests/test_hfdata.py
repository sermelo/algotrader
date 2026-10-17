import numpy as np
import pandas as pd
import pytest

from algotrader.hfdata import adjust_frame, adjustment_factor, align_events, detect_splits


def frame(closes, volumes=None, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": pd.Series(volumes if volumes is not None else 1000.0, index=idx)})


def test_split_scales_earlier_prices_only():
    raw = frame([100, 100, 50, 50])
    ev = pd.DataFrame({"split": [2.0], "dividend": [0.0]}, index=[raw.index[2]])
    adj = adjust_frame(raw, ev)
    assert adj["close"].tolist() == pytest.approx([50, 50, 50, 50])
    assert adj["volume"].tolist() == pytest.approx([2000, 2000, 1000, 1000])
    assert adj["raw_close"].tolist() == [100, 100, 50, 50]


def test_dividend_uses_previous_raw_close():
    raw = frame([100, 100, 98, 98])
    ev = pd.DataFrame({"split": [1.0], "dividend": [2.0]}, index=[raw.index[2]])
    adj = adjust_frame(raw, ev)
    assert adj["close"].tolist() == pytest.approx([98, 98, 98, 98])      # 100 * (1 - 2/100)
    assert adjust_frame(raw, ev, dividends=False)["close"].tolist() == [100, 100, 98, 98]


def test_split_and_dividend_compose_and_last_price_is_raw():
    raw = frame([200, 200, 100, 100, 99, 99])
    ev = pd.DataFrame({"split": [2.0, 1.0], "dividend": [0.0, 1.0]}, index=[raw.index[2], raw.index[4]])
    al = align_events(raw.index, ev)
    f = adjustment_factor(raw["close"], al["split"], al["dividend"])
    assert f.iloc[-1] == 1.0 and f.iloc[4] == 1.0
    assert f.iloc[3] == pytest.approx(0.99)
    assert f.iloc[0] == pytest.approx(0.99 / 2)


def test_event_on_missing_day_moves_to_next_bar_and_late_events_drop():
    raw = frame([100, 100, 50, 50])
    gap_day = raw.index[1] + pd.Timedelta(hours=1)      # between two bars
    ev = pd.DataFrame({"split": [2.0, 3.0], "dividend": [0.0, 0.0]},
                      index=[gap_day, raw.index[-1] + pd.Timedelta(days=5)])
    al = align_events(raw.index, ev)
    assert al["split"].tolist() == [1.0, 1.0, 2.0, 1.0]


def test_same_bar_events_combine_and_no_events_is_identity():
    raw = frame([10, 10, 10])
    ev = pd.DataFrame({"split": [2.0, 3.0], "dividend": [0.1, 0.2]}, index=[raw.index[1]] * 2)
    al = align_events(raw.index, ev)
    assert al.iloc[1].tolist() == pytest.approx([6.0, 0.3])
    assert adjust_frame(raw, None)["close"].tolist() == [10, 10, 10]
    assert adjust_frame(raw, pd.DataFrame({"split": [], "dividend": []}))["close"].tolist() == [10, 10, 10]


def test_detects_forward_and_reverse_splits_and_ignores_big_moves():
    n = 60
    closes = [100.0] * n
    vols = [1000.0] * n
    closes[30:] = [25.0] * (n - 30)                 # 4-for-1 on day 30, volume x4
    vols[30:45] = [4000.0] * 15
    closes[45:] = [250.0] * (n - 45)                # 1-for-10 reverse on day 45, volume /10
    vols[45:] = [100.0] * (n - 45)
    raw = frame(closes, vols)
    s = detect_splits(raw)
    assert list(s.index) == [raw.index[30], raw.index[45]]
    assert s.tolist() == pytest.approx([4.0, 0.1])


def test_gap_without_volume_jump_is_not_a_split():
    closes = [100.0] * 30 + [52.0] * 30
    raw = frame(closes, [1000.0] * 60)
    assert detect_splits(raw).empty


def test_detected_splits_adjust_to_a_smooth_series():
    closes = [100.0] * 30 + [25.0] * 30
    vols = [1000.0] * 30 + [4000.0] * 30
    raw = frame(closes, vols)
    s = detect_splits(raw)
    adj = adjust_frame(raw, pd.DataFrame({"split": s, "dividend": 0.0}))
    assert adj["close"].tolist() == pytest.approx([25.0] * 60)


def test_news_gap_with_a_fading_volume_spike_is_not_a_split():
    closes = [100.0] * 30 + [52.0] * 30
    vols = [1000.0] * 30 + [5000.0, 3000.0] + [1000.0] * 28
    assert detect_splits(frame(closes, vols)).empty


def test_gap_off_the_split_ratio_is_not_a_split():
    closes = [100.0] * 30 + [75.0] * 30                       # -25% is not 2/3 within 5%
    vols = [1000.0] * 30 + [1500.0] * 30
    assert detect_splits(frame(closes, vols)).empty


def test_exact_two_for_one_gap_needs_no_volume_jump_but_a_looser_one_does():
    exact = frame([100.0] * 30 + [50.0] * 30, [1000.0] * 60)
    assert detect_splits(exact).tolist() == [2.0]
    loose = frame([100.0] * 30 + [51.5] * 30, [1000.0] * 60)          # 3% off 1/2
    assert detect_splits(loose).empty


from algotrader.hfdata import build_series


def reference_for(raw, last=None, split=None):
    ref = pd.DataFrame({"raw_close": raw["close"], "split": 1.0, "dividend": 0.0})
    if split:
        for d, k in split.items():
            ref.loc[d, "split"] = k
    return ref if last is None else ref[ref.index <= last]


def test_build_with_matching_reference_uses_its_events_and_is_verified():
    raw = frame([100.0] * 80 + [50.0] * 80, [1000.0] * 80 + [2000.0] * 80)
    ref = reference_for(raw, split={raw.index[80]: 2.0})
    adj, info = build_series(raw, ref)
    assert info["status"] == "ok" and info["agree"] == 1.0
    assert adj["close"].tolist() == pytest.approx([50.0] * 160)
    assert adj["split"].iloc[80] == 2.0


def test_build_rejects_a_reused_ticker():
    raw = frame([100.0] * 100)
    other = reference_for(frame([30.0] * 100))
    adj, info = build_series(raw, other)
    assert adj is None and info["status"] == "mismatch"


def test_build_without_reference_detects_splits_and_is_unverified():
    raw = frame([100.0] * 80 + [50.0] * 80, [1000.0] * 80 + [2000.0] * 80)
    adj, info = build_series(raw)
    assert info["status"] == "unverified" and info["detected"] == 1
    assert adj["close"].tolist() == pytest.approx([50.0] * 160)


def test_build_takes_reference_events_then_detected_splits_after_its_last_day():
    closes = [100.0] * 60 + [50.0] * 60 + [25.0] * 60
    vols = [1000.0] * 60 + [2000.0] * 60 + [4000.0] * 60
    raw = frame(closes, vols)
    ref = reference_for(raw, last=raw.index[119], split={raw.index[60]: 2.0})
    adj, info = build_series(raw, ref)
    assert info["status"] == "ok" and info["detected"] == 1
    assert adj["close"].tolist() == pytest.approx([25.0] * 180)


def test_build_too_short_series_is_dropped():
    adj, info = build_series(frame([10.0] * 5))
    assert adj is None and info["status"] == "too_short"


def test_no_splits_gives_an_empty_series_on_a_date_index():
    s = detect_splits(frame([10.0] * 40))
    assert s.empty and isinstance(s.index, pd.DatetimeIndex)
    assert (s.index > pd.Timestamp("2020-01-01")).tolist() == []
