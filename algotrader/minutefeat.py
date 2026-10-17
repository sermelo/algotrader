"""Minute-bar features and forward returns for the small-account intraday models (roadmap S14).

One row per S&P 500 member, session and decision minute (10:00, 10:30, ..., 15:00 New York): features
known at that minute's close and the gross return of buying at the next bar's open and selling after
10, 20, 30, 90 or 240 minutes or at the close, with exactly the exit rule of `slot.SlotTrader` (the open of
the first bar at or after entry + h, else the last bar's close). Bars exist only for minutes with trades;
a price is the last trade so far (forward-filled), a decision needs a first trade and an entry needs a
trade in the entry minute. Sessions that closed at 13:00 are detected from SPY's after-hours activity and
cut there.

Features (all causal): returns over the last 5, 15, 30 and 60 minutes and since the open (`ret5` ...
`ret_open`), the opening gap and the previous day's return (`gap`, `prev_ret`), realised volatility of 1-minute
returns over 30 minutes in bp (`vol30`) and of daily returns over 20 days (`dvol20`), volume since the open and
over the last 30 minutes against the 20-day average (`rvol_day`, `rvol30`), distance from the session VWAP
(`vwap_dist`), position in the day's range (`range_pos`), a spread proxy: the median high-low range of the last
30 bars in bp (`range_bp`), bars with trades in the last 30 minutes (`active30`), SPY's returns and the stock's
returns against them (`spy_ret30`, `spy_ret_open`, `rel30`, `rel_open`), `minute` since the open, `mins_left` in the
session, `price` and the log 20-day dollar volume (`logdv20`).
"""
import warnings

import numpy as np
import pandas as pd

M = 390                                   # regular session minutes
EARLY_END = 210                           # a 13:00 close
DECISIONS = tuple(range(30, 331, 30))     # decision at the end of minute d - 1 (10:00 ... 15:00), entry bar d
HORIZONS = (10, 20, 30, 90, 240)
MARKET = "SPY"
FEATURES = ["ret5", "ret15", "ret30", "ret60", "ret_open", "gap", "prev_ret", "vol30", "dvol20", "rvol_day",
            "rvol30", "vwap_dist", "range_pos", "range_bp", "active30", "spy_ret30", "spy_ret_open", "rel30",
            "rel_open", "minute", "mins_left", "price", "logdv20"]
LABELS = [f"fwd{h}" for h in HORIZONS] + ["fwd_close"]


def early_close_days(spy: pd.DataFrame) -> set:
    """Days whose session ended at 13:00: SPY has almost no trades between 13:05 and 16:00 compared with
    the morning (after-hours prints only). `spy` has all hours, columns t (New York, naive) and volume."""
    minute = spy["t"].dt.hour * 60 + spy["t"].dt.minute
    day = spy["t"].dt.normalize()
    morning = day[(minute >= 570) & (minute < 780)].value_counts()
    late = day[(minute > 784) & (minute < 960)].value_counts()
    return {d for d, m in morning.items() if m > 0 and late.get(d, 0) < 0.2 * m}


def _ffill(x, has):
    """Forward fill along the last axis from the bars that exist (NaN before the first)."""
    idx = np.where(has, np.arange(x.shape[-1]), 0)
    idx = np.maximum.accumulate(idx, axis=-1)
    out = np.take_along_axis(x, idx, axis=-1)
    seen = np.maximum.accumulate(has, axis=-1)
    return np.where(seen, out, np.nan), seen


def _next_existing(has):
    """Index of the first existing bar at or after each minute (M if none)."""
    idx = np.where(has, np.arange(has.shape[-1]), has.shape[-1])
    return np.minimum.accumulate(idx[..., ::-1], axis=-1)[..., ::-1]


def daily_context(days, tickers, close, volume, carry):
    """Previous close, previous day's return, 20-day return volatility, 20-day average volume and log dollar
    volume for each (day, ticker), from the sessions before each day (`carry`: earlier days, long frame
    date/ticker/close/volume)."""
    cd = pd.DataFrame(close, index=days, columns=tickers)
    vd = pd.DataFrame(volume, index=days, columns=tickers)
    if carry is not None and len(carry):
        cc = carry.pivot(index="date", columns="ticker", values="close").reindex(columns=tickers)
        cv = carry.pivot(index="date", columns="ticker", values="volume").reindex(columns=tickers)
        cd, vd = pd.concat([cc, cd]), pd.concat([cv, vd])
    ret = cd.ffill().pct_change(fill_method=None)
    prev_close = cd.ffill().shift(1)
    out = {"prev_close": prev_close, "prev_ret": ret.shift(1),
           "dvol20": ret.shift(1).rolling(20, min_periods=5).std(),
           "avg_vol": vd.shift(1).rolling(20, min_periods=5).mean()}
    out["logdv20"] = np.log(out["avg_vol"] * prev_close)
    return {k: v.reindex(days).to_numpy(float) for k, v in out.items()}, cd, vd


def process(df: pd.DataFrame, members: pd.DataFrame, carry: pd.DataFrame = None):
    with np.errstate(all="ignore"):
        return _process(df, members, carry)


def _process(df, members, carry):
    """Features and labels for one stretch of sessions.

    df: minute bars (ticker, t = New York naive start of the bar, open, high, low, close, volume); SPY with
    all hours, the others regular session. members: bool (days x tickers) membership, rows only for days
    with bars (forward-filled by the caller). carry: the previous sessions' daily close and volume.
    Returns (rows, carry_out): one row per member, day and decision minute with `FEATURES` and `LABELS`.
    """
    spy_all = df[df["ticker"] == MARKET]
    early = early_close_days(spy_all) if len(spy_all) else set()
    minute = df["t"].dt.hour * 60 + df["t"].dt.minute - 570
    df = df[(minute >= 0) & (minute < M)].assign(m=lambda x: minute[x.index], day=lambda x: x["t"].dt.normalize())
    days = pd.DatetimeIndex(sorted(df["day"].unique()))
    tickers = sorted(df["ticker"].unique())
    if MARKET not in tickers:
        raise ValueError("SPY bars are needed for the market context")
    di, tj = days.get_indexer(df["day"]), pd.Index(tickers).get_indexer(df["ticker"])
    shape = (len(days), len(tickers), M)
    A = {k: np.full(shape, np.nan, dtype=np.float32) for k in ("open", "high", "low", "close")}
    vol = np.zeros(shape, dtype=np.float32)
    has = np.zeros(shape, dtype=bool)
    mm = df["m"].to_numpy()
    for k in A:
        A[k][di, tj, mm] = df[k].to_numpy(np.float32)
    vol[di, tj, mm] = df["volume"].to_numpy(np.float32)
    has[di, tj, mm] = True
    end = np.array([EARLY_END if d in early else M for d in days])
    has &= np.arange(M)[None, None, :] < end[:, None, None]                      # nothing after an early close
    for k in A:
        A[k] = np.where(has, A[k], np.nan)
    vol = np.where(has, vol, 0.0)
    close_f, seen = _ffill(A["close"], has)
    nxt = _next_existing(has)
    open_bf = np.take_along_axis(np.concatenate([A["open"], np.full(shape[:2] + (1,), np.nan, np.float32)], -1), nxt, -1)
    day_close = close_f[:, :, -1]
    day_vol = vol.sum(-1)
    traded = has.any(-1)
    ctx, cd, vd = daily_context(days, tickers, np.where(traded, day_close, np.nan), np.where(traded, day_vol, np.nan), carry)
    avg_vol = ctx["avg_vol"]
    cum_v = np.cumsum(vol, -1)
    tp = np.where(has, (np.nan_to_num(A["high"]) + np.nan_to_num(A["low"]) + np.nan_to_num(A["close"])) / 3, 0.0)
    cum_pv = np.cumsum(tp * vol, -1)
    run_hi = np.maximum.accumulate(np.where(has, A["high"], -np.inf), -1)
    run_lo = np.minimum.accumulate(np.where(has, A["low"], np.inf), -1)
    lr = np.diff(np.log(close_f), axis=-1, prepend=np.nan)
    lr = np.nan_to_num(lr)
    s1, s2 = np.cumsum(lr, -1), np.cumsum(lr * lr, -1)
    rng_bp = np.where(has, (A["high"] - A["low"]) / A["close"] * 1e4, np.nan)
    open_day = open_bf[:, :, 0]
    spy = tickers.index(MARKET)
    member = members.reindex(index=days, columns=tickers, fill_value=False).to_numpy(bool)
    out = []
    for d in DECISIONS:
        k = d - 1
        c = close_f[:, :, k]
        ok = seen[:, :, k] & member & (c > 0)
        feat = {}
        for n in (5, 15, 30, 60):
            feat[f"ret{n}"] = c / close_f[:, :, k - n] - 1 if k - n >= 0 else np.full_like(c, np.nan)
        feat["ret_open"] = c / open_day - 1
        feat["gap"] = open_day / ctx["prev_close"] - 1
        feat["prev_ret"] = ctx["prev_ret"]
        w = slice(k - 29, k + 1) if k >= 29 else slice(0, k + 1)
        n = k + 1 if k < 29 else 30
        a, b = (s1[:, :, k] - (s1[:, :, k - 30] if k >= 30 else 0)), (s2[:, :, k] - (s2[:, :, k - 30] if k >= 30 else 0))
        feat["vol30"] = np.sqrt(np.maximum(b - a * a / n, 0) / max(n - 1, 1)) * 1e4
        feat["dvol20"] = ctx["dvol20"]
        feat["rvol_day"] = cum_v[:, :, k] / avg_vol
        v30 = cum_v[:, :, k] - (cum_v[:, :, k - 30] if k >= 30 else 0)
        feat["rvol30"] = v30 * (M / 30) / avg_vol
        feat["vwap_dist"] = c / (cum_pv[:, :, k] / np.where(cum_v[:, :, k] > 0, cum_v[:, :, k], np.nan)) - 1
        hi, lo = run_hi[:, :, k], run_lo[:, :, k]
        feat["range_pos"] = np.where(hi > lo, (c - lo) / (hi - lo), 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat["range_bp"] = np.nanmedian(rng_bp[:, :, w], axis=-1)
        feat["active30"] = has[:, :, w].sum(-1).astype(float)
        feat["spy_ret30"] = np.repeat(feat["ret30"][:, spy:spy + 1], len(tickers), 1)
        feat["spy_ret_open"] = np.repeat(feat["ret_open"][:, spy:spy + 1], len(tickers), 1)
        feat["rel30"] = feat["ret30"] - feat["spy_ret30"]
        feat["rel_open"] = feat["ret_open"] - feat["spy_ret_open"]
        feat["minute"] = np.full_like(c, d, dtype=float)
        feat["mins_left"] = np.repeat((end - d)[:, None], len(tickers), 1).astype(float)
        feat["price"] = c
        feat["logdv20"] = ctx["logdv20"]
        entry_ok = has[:, :, d] if d < M else np.zeros_like(ok)
        entry_ok = entry_ok & (np.asarray(end)[:, None] > d)
        entry = A["open"][:, :, d] if d < M else np.full_like(c, np.nan)
        lab = {}
        for h in HORIZONS:
            t = d + h
            ex = open_bf[:, :, t] if t < M else np.full_like(c, np.nan)     # open of the first bar at or after d + h
            ex = np.where(np.isnan(ex), day_close, ex)                       # no such bar: the last close
            lab[f"fwd{h}"] = ex / entry - 1
        lab["fwd_close"] = day_close / entry - 1
        keep = ok & entry_ok
        di_, tj_ = np.nonzero(keep)
        frame = pd.DataFrame({"date": days[di_], "ticker": np.asarray(tickers)[tj_]})
        for name in FEATURES:
            frame[name] = np.asarray(feat[name], dtype=np.float32)[di_, tj_]
        for name in LABELS:
            frame[name] = np.asarray(lab[name], dtype=np.float32)[di_, tj_]
        out.append(frame)
    rows = pd.concat(out, ignore_index=True)
    carry_out = pd.concat([cd.stack().rename("close"), vd.stack().rename("volume")], axis=1).dropna(subset=["close"])
    carry_out = carry_out.reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
    last_days = sorted(carry_out["date"].unique())[-25:]
    return rows.sort_values(["date", "minute", "ticker"], ignore_index=True), carry_out[carry_out["date"].isin(last_days)]


def membership_by_day(members_long: pd.DataFrame, days, tickers) -> pd.DataFrame:
    """Bool (days x tickers): S&P 500 member on each day, the latest list on or before it."""
    m = members_long.assign(v=True).pivot_table(index="date", columns="ticker", values="v", aggfunc="any", fill_value=False)
    m = m.reindex(columns=tickers, fill_value=False).astype(float)
    m = m.reindex(index=m.index.union(days)).ffill().reindex(days).fillna(0.0) > 0.5
    return m
