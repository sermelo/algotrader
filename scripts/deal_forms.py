"""Index the merger-related filings in SEC's submissions.zip (for `algotrader.mergerarb`).

Writes data/edgar/deal_filings.parquet: cik, form, filed, accepted, accession, doc, items for merger proxies
and information statements, tender-offer filings, DEFA14A, 25-NSE and 8-Ks with item 1.01 or 1.02, and prints
the count per form and year. Run on the VPS after `python -m algotrader.edgar download`.
"""
import json
import zipfile

import pandas as pd

FORMS = {"PREM14A", "DEFM14A", "DEFA14A", "SC 14D9", "SC14D9C", "SC TO-T", "SC TO-C", "PREM14C", "DEFM14C",
         "25-NSE", "8-K"}


def main(zip_path="data/edgar/submissions.zip", out="data/edgar/deal_filings.parquet"):
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.startswith("CIK"):
                continue
            d = json.loads(z.read(n))
            a = d["filings"]["recent"] if "filings" in d else d
            cik = int(n[3:13])
            forms = a.get("form", [])
            items = a.get("items") or [""] * len(forms)
            accepted = a.get("acceptanceDateTime") or [""] * len(forms)
            docs = a.get("primaryDocument") or [""] * len(forms)
            for i, form in enumerate(forms):
                if form not in FORMS:
                    continue
                if form == "8-K" and not ("1.01" in (items[i] or "") or "1.02" in (items[i] or "")):
                    continue
                rows.append((cik, form, a["filingDate"][i], accepted[i], a["accessionNumber"][i], docs[i],
                             items[i] or ""))
    df = pd.DataFrame(rows, columns=["cik", "form", "filed", "accepted", "accession", "doc", "items"])
    df.to_parquet(out)
    df["year"] = df["filed"].str[:4]
    print(df[df.form != "8-K"].pivot_table(index="year", columns="form", values="cik", aggfunc="count")
          .fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
