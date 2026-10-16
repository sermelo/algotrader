"""Fama-French factor returns (Kenneth French data library) and factor regressions.

Downloads the daily five factors (market, size, value, profitability, investment) and
the momentum factor, and regresses each experiment's walk-forward daily returns on
them, to see whether an "alpha" against buy and hold is just exposure to known factors.

Not used by the experiment runner: it reads the committed `*_wf_returns.csv` files.

Example:
    python -m algotrader.french download              # -> data/french/
    python -m algotrader.french report                # -> results/factor_alpha.md (+ broad)
"""
import argparse
import hashlib
import math
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "F-F_Momentum_Factor_daily_CSV.zip")
FACTORS = ("Mkt", "SMB", "HML", "RMW", "CMA", "Mom")
TRADING_DAYS = 252


def download(out_dir="data/french", opener=urllib.request.urlopen) -> list:
    """Fetch the factor files (zipped CSV) into `out_dir`; returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "algotrader/0.1"})
        with opener(req) as resp:
            data = resp.read()
        if not data.startswith(b"PK"):
            raise RuntimeError(f"{name}: response is not a zip file")
        (out / name).write_bytes(data)
        paths.append(out / name)
    return paths


def parse(text: str) -> pd.DataFrame:
    """The daily table of a French CSV: rows `YYYYMMDD, v1, v2, ...` in percent below a
    header row starting with a comma; returned as decimal returns by date."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and l.strip(", "))
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for line in lines[start + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if not (len(parts[0]) == 8 and parts[0].isdigit()):
            break
        rows.append([parts[0]] + [float(p) for p in parts[1:len(cols) + 1]])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    df.index.name = None
    return df / 100.0


def _read_zip(path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin1")


def load(data_dir="data/french") -> pd.DataFrame:
    """Daily factors: Mkt (total market return, Mkt-RF + RF), SMB, HML, RMW, CMA, Mom, RF.

    The market factor is the total return, not the excess return, because the
    backtests earn no interest on cash: a strategy holding beta of the market and
    the rest in cash then has zero alpha.
    """
    d = Path(data_dir)
    ff5 = parse(_read_zip(d / FILES[0]))
    mom = parse(_read_zip(d / FILES[1]))
    mom.columns = ["Mom"]
    df = ff5.join(mom, how="inner")
    df["Mkt"] = df["Mkt-RF"] + df["RF"]
    return df[list(FACTORS) + ["RF"]]


def factor_regression(r: pd.Series, factors: pd.DataFrame, cols=FACTORS) -> dict:
    """OLS of daily returns on the factor columns (common days only).

    Returns annualised alpha (intercept * 252), its t-statistic, each factor loading,
    R squared and the number of days.
    """
    df = pd.concat([pd.Series(r, name="_r"), factors[list(cols)]], axis=1, join="inner").dropna()
    n, k = len(df), len(cols)
    out = {"days": n}
    if n <= k + 2 or float(df["_r"].var()) == 0:
        return out | {"alpha": float("nan"), "alpha_t": float("nan"), "r2": float("nan")} | \
            {c: float("nan") for c in cols}
    y = df["_r"].to_numpy()
    X = np.column_stack([np.ones(n), df[list(cols)].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (n - k - 1)
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    out |= {"alpha": float(coef[0] * TRADING_DAYS),
            "alpha_t": float(coef[0] / se) if se > 0 else float("nan"),
            "r2": float(1 - resid.var() / y.var())}
    out |= {c: float(b) for c, b in zip(cols, coef[1:])}
    return out


def regress_results(results_dir, factors: pd.DataFrame, column="base") -> pd.DataFrame:
    """One row per `*_wf_returns.csv` in `results_dir` (the `column` cost variant),
    plus a `buy_and_hold` row from the files' `hold` column."""
    rows, hold = [], None
    for path in sorted(Path(results_dir).glob("*_wf_returns.csv")):
        wf = pd.read_csv(path, index_col=0, parse_dates=True)
        if hold is None and "hold" in wf:
            hold = wf["hold"]
        rows.append({"experiment": path.name.removesuffix("_wf_returns.csv")}
                    | factor_regression(wf[column], factors))
    if hold is not None:
        rows.append({"experiment": "buy_and_hold"} | factor_regression(hold, factors))
    return pd.DataFrame(rows).set_index("experiment") if rows else pd.DataFrame()


def _fmt(table: pd.DataFrame) -> str:
    head = "| Experiment | Alpha/yr (t) | " + " | ".join(FACTORS) + " | R² | Days |"
    lines = [head, "|" + "---|" * (len(FACTORS) + 4)]
    for name, r in table.sort_values("alpha_t", ascending=False).iterrows():
        if math.isnan(r["alpha"]):
            lines.append(f"| `{name}` | no returns | " + " | ".join("" for _ in FACTORS)
                         + f" | | {int(r['days'])} |")
            continue
        loads = " | ".join(f"{r[c]:.2f}" for c in FACTORS)
        lines.append(f"| `{name}` | {r['alpha']:+.1%} ({r['alpha_t']:.2f}) | {loads} | "
                     f"{r['r2']:.2f} | {int(r['days'])} |")
    return "\n".join(lines)


def write_report(results_dir, factors: pd.DataFrame, data_dir="data/french") -> Path:
    """Write `factor_alpha.csv` and `factor_alpha.md` into `results_dir`."""
    results_dir = Path(results_dir)
    table = regress_results(results_dir, factors)
    table.to_csv(results_dir / "factor_alpha.csv", float_format="%.6g")
    files = "; ".join(f"`{n}` sha256 {hashlib.sha256((Path(data_dir) / n).read_bytes()).hexdigest()[:12]}"
                      for n in FILES if (Path(data_dir) / n).exists())
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    last = factors.index.max()
    text = f"""# Factor regressions

Generated by `python -m algotrader.french report`; do not edit by hand. Walk-forward
daily returns (base costs) regressed on the Fama-French five factors and momentum
(Kenneth French data library, daily). Mkt is the total market return (excess return +
risk-free rate), because the backtests earn no interest on cash. Only days present in
both series are used (the factor file ends {last:%Y-%m-%d}). Alpha is the intercept
times 252 with its OLS t-statistic; the other columns are factor loadings.

{_fmt(table)}

Factor files: {files}. Code: `french.py` sha256 {code}.
"""
    path = results_dir / "factor_alpha.md"
    path.write_text(text)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["download", "report"])
    p.add_argument("--data-dir", default="data/french")
    p.add_argument("--results", nargs="*", default=["results", "results/broad"])
    args = p.parse_args(argv)
    if args.action == "download":
        for path in download(args.data_dir):
            print("saved", path)
        return
    factors = load(args.data_dir)
    for d in args.results:
        if Path(d).is_dir():
            print("wrote", write_report(d, factors, args.data_dir))


if __name__ == "__main__":
    main()
