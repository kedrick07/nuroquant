"""
Final ticker selection: walk down a market-cap-ranked candidate pool (e.g. top 80),
combine FNSPID + NYT monthly article counts per ticker, apply the coverage gate
(avg >=5 articles/month AND >=85% of months with >=1 article), and keep selecting
down the ranked list until you have --target-n tickers that pass.

Inputs:
  --candidates   data/universe/candidates.csv
                 columns: ticker,company_name,market_cap_rank  (rank 1 = largest, ranked over 2019-2023)
                 -> this file should now have your POOL size (e.g. 80 rows), not just 30
  --fnspid       data/news/fnspid/processed/fnspid_monthly_counts.csv   (from script 1)
  --nyt          data/news/nytarchive/processed/nytarchive_monthly_counts.csv (from script 4)

Output:
  data/universe/final_selection.csv       -> only the tickers that passed, in rank order, capped at --target-n
  data/universe/screening_report.csv      -> every candidate screened, pass/fail + metrics (for the thesis appendix)

Usage:
  python 5_coverage_gate_selection.py --candidates data/universe/candidates.csv --target-n 30
"""
import argparse
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/universe")


def build_month_grid(start: str, end: str):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/universe/candidates.csv")
    ap.add_argument("--fnspid", default="data/news/fnspid/processed/fnspid_monthly_counts.csv")
    ap.add_argument("--nyt", default="data/news/nytarchive/processed/nytarchive_monthly_counts.csv")
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--target-n", type=int, default=30, help="final number of tickers you need to lock")
    ap.add_argument("--min-avg", type=float, default=5.0)
    ap.add_argument("--min-pct-covered", type=float, default=85.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    months = build_month_grid(args.start, args.end)
    n_months = len(months)

    candidates = pd.read_csv(args.candidates)
    candidates["ticker"] = candidates["ticker"].str.upper().str.strip()
    candidates = candidates.sort_values("market_cap_rank")

    fnspid = pd.read_csv(args.fnspid) if Path(args.fnspid).exists() else pd.DataFrame(columns=["ticker", "year_month", "article_count"])
    nyt = pd.read_csv(args.nyt) if Path(args.nyt).exists() else pd.DataFrame(columns=["ticker", "year_month", "article_count"])
    combined = pd.concat([fnspid, nyt], ignore_index=True)
    combined["ticker"] = combined["ticker"].str.upper().str.strip()
    combined = combined.groupby(["ticker", "year_month"], as_index=False)["article_count"].sum()

    report_rows = []
    selected = []

    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        g = combined[combined["ticker"] == ticker]
        month_map = dict(zip(g["year_month"], g["article_count"]))
        vals = [month_map.get(m, 0) for m in months]

        avg = sum(vals) / n_months
        pct_covered = sum(1 for v in vals if v >= 1) / n_months * 100
        passed = (avg >= args.min_avg) and (pct_covered >= args.min_pct_covered)

        report_rows.append({
            "market_cap_rank": row["market_cap_rank"],
            "ticker": ticker,
            "company_name": row.get("company_name", ""),
            "avg_articles_per_month": round(avg, 2),
            "pct_months_covered": round(pct_covered, 1),
            "total_articles": sum(vals),
            "passed_gate": passed,
            "selected": False,
        })

        if passed and len(selected) < args.target_n:
            selected.append(ticker)
            report_rows[-1]["selected"] = True

    report = pd.DataFrame(report_rows)
    report.to_csv(OUT_DIR / "screening_report.csv", index=False)

    final = report[report["selected"]].reset_index(drop=True)
    final.to_csv(OUT_DIR / "final_selection.csv", index=False)

    print(f"Candidate pool screened: {len(report)}")
    print(f"Passed coverage gate: {report['passed_gate'].sum()}")
    print(f"Selected (target={args.target_n}): {len(final)}")
    if len(final) < args.target_n:
        print(f"\nWARNING: only found {len(final)}/{args.target_n} passing tickers. "
              f"Widen your candidate pool (more than {len(candidates)} rows) and re-run.")
    print(f"\n-> {OUT_DIR / 'final_selection.csv'}")
    print(f"-> {OUT_DIR / 'screening_report.csv'} (full audit trail incl. rejected tickers)")


if __name__ == "__main__":
    main()
