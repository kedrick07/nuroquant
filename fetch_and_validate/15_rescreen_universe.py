"""
To-do 8b: Re-screen the universe on the CLEANED coverage numbers and build final_30_tickers_v2.csv.

Locks the tickers from the current 30 that meet --lock-min-pct month coverage (or are listed in
--exceptions), then walks the market-cap-ranked candidate list and fills the remaining slots with the
first candidates whose FNSPID-only month coverage is >= --min-pct and average >= --min-avg.

Candidates are screened on FNSPID only (no NYT), which is conservative.

Inputs : data/news/news_coverage_report.csv        (from #8, the current 30)
         data/universe/candidates.csv              (market-cap ranked candidates)
         data/news/fnspid/processed/fnspid_monthly_counts.csv  (from script 1)
Outputs: data/universe/final_30_tickers_v2.csv
         data/universe/rescreen_report.csv

Usage (project root):
  python fetch_and_validate/15_rescreen_universe.py
  python fetch_and_validate/15_rescreen_universe.py --exceptions NVDA,MSFT
  python fetch_and_validate/15_rescreen_universe.py --lock-min-pct 95
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/universe/candidates.csv")
    ap.add_argument("--fnspid-counts", default="data/news/fnspid/processed/fnspid_monthly_counts.csv")
    ap.add_argument("--coverage-report", default="data/news/news_coverage_report.csv")
    ap.add_argument("--exceptions", default="NVDA")
    ap.add_argument("--exclude", default="GOOGL", help="same-company duplicates, e.g. a second share class")
    ap.add_argument("--lock-min-pct", type=float, default=100.0)
    ap.add_argument("--min-pct", type=float, default=100.0)
    ap.add_argument("--min-avg", type=float, default=5.0)
    ap.add_argument("--target-n", type=int, default=30)
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--out", default="data/universe/final_30_tickers_v2.csv")
    ap.add_argument("--report", default="data/universe/rescreen_report.csv")
    args = ap.parse_args()

    exceptions = {t.strip().upper() for t in args.exceptions.split(",") if t.strip()}
    exclude = {t.strip().upper() for t in args.exclude.split(",") if t.strip()}
    months = [p.strftime("%Y-%m") for p in pd.period_range(args.start, args.end, freq="M")]

    cov = pd.read_csv(args.coverage_report)
    cov["ticker"] = cov["ticker"].str.upper()
    keep_mask = (cov["pct_months_covered"] >= args.lock_min_pct) | cov["ticker"].isin(exceptions)
    locked = cov.loc[keep_mask, "ticker"].tolist()
    swapped_out = cov.loc[~keep_mask, "ticker"].tolist()
    print(f"Locked from current list: {len(locked)} | swapped out: {swapped_out}")
    need = args.target_n - len(locked)

    cands = pd.read_csv(args.candidates)
    cands["ticker"] = cands["ticker"].str.upper().str.strip()
    cands = cands.sort_values("market_cap_rank").reset_index(drop=True)

    fn = pd.read_csv(args.fnspid_counts)
    fn["ticker"] = fn["ticker"].astype(str).str.upper().str.strip()
    piv = fn.pivot_table(index="ticker", columns="year_month", values="article_count", aggfunc="sum", fill_value=0)
    piv = piv.reindex(columns=months, fill_value=0)

    rows, picked = [], []
    skip = set(locked) | set(swapped_out) | exclude
    for _, r in cands.iterrows():
        t = r["ticker"]
        if t in skip:
            continue
        vec = piv.loc[t] if t in piv.index else pd.Series(0, index=months)
        pct = (vec >= 1).mean() * 100
        avg = vec.sum() / len(months)
        ok = (pct >= args.min_pct) and (avg >= args.min_avg)
        take = ok and len(picked) < need
        if take:
            picked.append(t)
        rows.append({"market_cap_rank": r["market_cap_rank"], "ticker": t, "company_name": r.get("company_name", ""),
                     "fnspid_pct_months": round(pct, 1), "fnspid_avg_per_month": round(avg, 1),
                     "months_under_3": int((vec < 3).sum()), "passes": ok, "selected": take})
        if len(picked) >= need:
            break

    rep = pd.DataFrame(rows)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.report, index=False)

    names = cands.set_index("ticker")
    final = pd.DataFrame({"ticker": locked + picked})
    final["market_cap_rank"] = final["ticker"].map(names["market_cap_rank"])
    final["company_name"] = final["ticker"].map(names["company_name"])
    final["source"] = ["locked"] * len(locked) + ["replacement"] * len(picked)
    final["notes"] = final["ticker"].map(lambda t: "kept as explicit coverage exception" if t in exceptions and
                                         cov.loc[cov.ticker == t, "pct_months_covered"].iloc[0] < args.lock_min_pct else "")
    final = final.sort_values("market_cap_rank").reset_index(drop=True)
    final.insert(0, "selection_rank", range(1, len(final) + 1))
    final.to_csv(args.out, index=False)

    print(f"\nReplacements needed: {need} | found: {len(picked)}")
    print(rep.to_string(index=False))
    if len(picked) < need:
        print(f"\nWARNING: only {len(picked)}/{need} replacements found; relax --min-pct or --min-avg")
    print(f"\nFinal list ({len(final)}) -> {args.out}")
    print(final[["selection_rank", "market_cap_rank", "ticker", "company_name", "source", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
