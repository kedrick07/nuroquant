"""
Match raw NYT Archive articles to tickers by company name, and compute per-ticker
monthly coverage (parallel to script 1's FNSPID output). Run this AFTER script 2
has fetched the raw months and you have a candidate ticker/company list.

Requires a CSV at data/universe/candidates.csv with columns:
  ticker,company_name,aliases
  AAPL,Apple Inc,Apple
  MSFT,Microsoft Corp,Microsoft
  ...
("aliases" is optional, pipe-separated extra names/short forms to match on.)

Matching logic: for each article, search headline.main + abstract + snippet +
lead_paragraph + keywords[].value (case-insensitive) for the company name or any alias.

Output: data/news/nytarchive/processed/nytarchive_monthly_counts.csv
        data/news/nytarchive/processed/nytarchive_summary.csv

Usage:
  python 4_nytarchive_ticker_coverage.py --candidates data/universe/candidates.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/news/nytarchive/raw")
OUT_DIR = Path("data/news/nytarchive/processed")


def load_candidates(path: str):
    df = pd.read_csv(path)
    entries = []
    for _, row in df.iterrows():
        names = [str(row["company_name"]).strip()]
        if "aliases" in df.columns and pd.notna(row.get("aliases")):
            names += [a.strip() for a in str(row["aliases"]).split("|") if a.strip()]
        patterns = [re.compile(re.escape(n), re.IGNORECASE) for n in names if n]
        entries.append((str(row["ticker"]).upper().strip(), patterns))
    return entries


def article_text(doc: dict) -> str:
    parts = [
        doc.get("headline", {}).get("main", "") or "",
        doc.get("abstract", "") or "",
        doc.get("snippet", "") or "",
        doc.get("lead_paragraph", "") or "",
        " ".join(kw.get("value", "") for kw in doc.get("keywords", []) if kw.get("value")),
    ]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/universe/candidates.csv")
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_candidates(args.candidates)
    months = pd.period_range(args.start, args.end, freq="M")

    records = []
    for p in months:
        year, month = p.year, p.month
        fpath = RAW_DIR / f"{year}-{month:02d}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        docs = data.get("response", {}).get("docs", [])
        year_month = f"{year}-{month:02d}"

        texts = [article_text(d) for d in docs]
        for ticker, patterns in tickers:
            count = sum(1 for t in texts if any(p.search(t) for p in patterns))
            if count:
                records.append({"ticker": ticker, "year_month": year_month, "article_count": count})

    counts = pd.DataFrame(records)
    counts.to_csv(OUT_DIR / "nytarchive_monthly_counts.csv", index=False)
    print(f"Wrote {OUT_DIR / 'nytarchive_monthly_counts.csv'} ({len(counts)} rows)")

    all_months = [p.strftime("%Y-%m") for p in months]
    n_months = len(all_months)
    summary_rows = []
    for ticker, _ in tickers:
        g = counts[counts["ticker"] == ticker]
        month_map = dict(zip(g["year_month"], g["article_count"]))
        vals = [month_map.get(m, 0) for m in all_months]
        avg = sum(vals) / n_months
        pct_covered = sum(1 for v in vals if v >= 1) / n_months * 100
        summary_rows.append({
            "ticker": ticker,
            "avg_articles_per_month": round(avg, 2),
            "pct_months_covered": round(pct_covered, 1),
            "total_articles": sum(vals),
        })

    summary = pd.DataFrame(summary_rows).sort_values("avg_articles_per_month", ascending=False)
    summary.to_csv(OUT_DIR / "nytarchive_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'nytarchive_summary.csv'} ({len(summary)} tickers)")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
