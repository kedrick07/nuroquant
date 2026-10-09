"""
To-do #2 helper: how often are FNSPID text fields actually filled for the final 30 tickers?

Streams both FNSPID CSVs (chunked), keeps only final-30 tickers within 2019-2023,
and reports the share of rows with non-empty Article_title, Lsa_summary, Article,
plus median length in characters. Output: data/news/fnspid/processed/text_field_fill_rates.csv

Usage:
  python 8_check_fnspid_text_fields.py --tickers data/universe/final_30_tickers.csv
"""
import argparse
from pathlib import Path

import pandas as pd

FNSPID_DIR = Path("data/news/fnspid")
FILES = ["All_external.csv", "nasdaq_exteral_data.csv"]
OUT = FNSPID_DIR / "processed" / "text_field_fill_rates.csv"
TEXT_COLS = ["Article_title", "Lsa_summary", "Article"]
CHUNK = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="data/universe/final_30_tickers.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    tickers = set(pd.read_csv(args.tickers)["ticker"].str.upper().str.strip())
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end) + pd.Timedelta(days=1)

    stats = {}
    for fname in FILES:
        path = FNSPID_DIR / fname
        if not path.exists():
            print(f"[skip] {path}")
            continue
        print(f"Scanning {fname} ...")
        header = pd.read_csv(path, nrows=0).columns
        cols = ["Date", "Stock_symbol"] + [c for c in TEXT_COLS if c in header]
        missing = [c for c in TEXT_COLS if c not in header]
        if missing:
            print(f"  columns not found in {fname}: {missing}")
        for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
            chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper().str.strip()
            chunk = chunk[chunk["Stock_symbol"].isin(tickers)]
            if chunk.empty:
                continue
            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
            chunk = chunk[(chunk["Date"] >= start) & (chunk["Date"] < end)]
            for col in TEXT_COLS:
                if col not in chunk.columns:
                    chunk[col] = pd.NA
            for t, g in chunk.groupby("Stock_symbol"):
                s = stats.setdefault(t, {"rows": 0, **{f"{c}_filled": 0 for c in TEXT_COLS},
                                         **{f"{c}_len": [] for c in TEXT_COLS}})
                s["rows"] += len(g)
                for c in TEXT_COLS:
                    txt = g[c].astype(str).str.strip()
                    ok = g[c].notna() & (txt != "") & (txt.str.lower() != "nan")
                    s[f"{c}_filled"] += int(ok.sum())
                    s[f"{c}_len"].append(int(txt[ok].str.len().median()) if ok.any() else 0)

    rows = []
    for t, s in sorted(stats.items()):
        r = {"ticker": t, "rows": s["rows"]}
        for c in TEXT_COLS:
            r[f"{c}_fill_pct"] = round(100 * s[f"{c}_filled"] / s["rows"], 1) if s["rows"] else 0
            r[f"{c}_median_chars"] = int(pd.Series(s[f"{c}_len"]).median()) if s[f"{c}_len"] else 0
        rows.append(r)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    total = sum(s["rows"] for s in stats.values())
    print(f"\nFinal-30 FNSPID rows in window: {total:,} across {len(stats)} tickers")
    for c in TEXT_COLS:
        filled = sum(s[f"{c}_filled"] for s in stats.values())
        print(f"  {c}: {100*filled/total:.1f}% non-empty")
    print(f"\nPer-ticker table -> {OUT}")
    print(df.sort_values("Lsa_summary_fill_pct").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
