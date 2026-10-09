"""
Analyze FNSPID news coverage per ticker per month.

Reads data/news/fnspid/All_external.csv and data/news/fnspid/nasdaq_exteral_data.csv
(large files -> streamed in chunks), filters to STUDY_START..STUDY_END, and produces:
  - data/news/fnspid/processed/fnspid_monthly_counts.csv   (ticker, year_month, article_count)
  - data/news/fnspid/processed/fnspid_summary.csv          (ticker, avg_articles_per_month, pct_months_covered)

Usage:
  python 1_analyze_fnspid_coverage.py
  python 1_analyze_fnspid_coverage.py --start 2019-01 --end 2023-12
"""
import argparse
import os
from pathlib import Path

import pandas as pd

FNSPID_DIR = Path("data/news/fnspid")
OUT_DIR = FNSPID_DIR / "processed"
FILES = ["All_external.csv", "nasdaq_exteral_data.csv"]
CHUNK_SIZE = 250_000


def month_range(start: str, end: str):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def process_file(path: Path, start: str, end: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[skip] {path} not found")
        return pd.DataFrame(columns=["Stock_symbol", "year_month"])

    counts = []
    start_ts = pd.Timestamp(start + "-01")
    end_ts = pd.Timestamp(end + "-01") + pd.offsets.MonthEnd(1)

    usecols_candidates = ["Date", "Stock_symbol"]
    for chunk in pd.read_csv(
        path,
        usecols=lambda c: c in usecols_candidates,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk = chunk.dropna(subset=["Date", "Stock_symbol"])
        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        chunk = chunk.dropna(subset=["Date"])
        chunk = chunk[(chunk["Date"] >= start_ts) & (chunk["Date"] <= end_ts)]
        if chunk.empty:
            continue
        chunk["year_month"] = chunk["Date"].dt.strftime("%Y-%m")
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper().str.strip()
        grp = chunk.groupby(["Stock_symbol", "year_month"]).size().reset_index(name="article_count")
        counts.append(grp)

    if not counts:
        return pd.DataFrame(columns=["Stock_symbol", "year_month", "article_count"])

    out = pd.concat(counts, ignore_index=True)
    out = out.groupby(["Stock_symbol", "year_month"], as_index=False)["article_count"].sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_months = month_range(args.start, args.end)

    frames = []
    for fname in FILES:
        print(f"Processing {fname} ...")
        df = process_file(FNSPID_DIR / fname, args.start, args.end)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby(["Stock_symbol", "year_month"], as_index=False)["article_count"].sum()
    merged = merged.rename(columns={"Stock_symbol": "ticker"})
    merged.to_csv(OUT_DIR / "fnspid_monthly_counts.csv", index=False)
    print(f"Wrote {OUT_DIR / 'fnspid_monthly_counts.csv'} ({len(merged)} rows)")

    n_months = len(all_months)
    summary_rows = []
    for ticker, g in merged.groupby("ticker"):
        month_map = dict(zip(g["year_month"], g["article_count"]))
        counts = [month_map.get(m, 0) for m in all_months]
        avg = sum(counts) / n_months
        pct_covered = sum(1 for c in counts if c >= 1) / n_months * 100
        summary_rows.append({
            "ticker": ticker,
            "avg_articles_per_month": round(avg, 2),
            "pct_months_covered": round(pct_covered, 1),
            "total_articles": sum(counts),
        })

    summary = pd.DataFrame(summary_rows).sort_values("avg_articles_per_month", ascending=False)
    summary.to_csv(OUT_DIR / "fnspid_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'fnspid_summary.csv'} ({len(summary)} tickers)")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
