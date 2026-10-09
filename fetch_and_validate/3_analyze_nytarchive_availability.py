"""
Analyze raw NYT Archive availability (data completeness check, not ticker matching yet).

Checks which months in the study window have been fetched to data/news/nytarchive/raw/,
counts articles per month, and flags gaps/failures.

Output: data/news/nytarchive/processed/nytarchive_availability.csv

Usage:
  python 3_analyze_nytarchive_availability.py --start 2019-01 --end 2023-12
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/news/nytarchive/raw")
OUT_DIR = Path("data/news/nytarchive/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(args.start, args.end, freq="M")

    rows = []
    for p in months:
        year, month = p.year, p.month
        fpath = RAW_DIR / f"{year}-{month:02d}.json"
        if not fpath.exists():
            rows.append({"year_month": f"{year}-{month:02d}", "status": "missing", "n_articles": 0})
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            docs = data.get("response", {}).get("docs", [])
            rows.append({"year_month": f"{year}-{month:02d}", "status": "ok", "n_articles": len(docs)})
        except Exception as e:
            rows.append({"year_month": f"{year}-{month:02d}", "status": f"corrupt: {e}", "n_articles": 0})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "nytarchive_availability.csv", index=False)

    n_total = len(df)
    n_ok = (df["status"] == "ok").sum()
    n_missing = (df["status"] == "missing").sum()
    n_corrupt = n_total - n_ok - n_missing

    print(f"Months in window: {n_total}")
    print(f"Fetched OK: {n_ok}  |  Missing: {n_missing}  |  Corrupt: {n_corrupt}")
    print(f"Total articles across fetched months: {df['n_articles'].sum():,}")
    if n_missing or n_corrupt:
        print("\nProblem months:")
        print(df[df["status"] != "ok"].to_string(index=False))
    print(f"\nFull report -> {OUT_DIR / 'nytarchive_availability.csv'}")


if __name__ == "__main__":
    main()
