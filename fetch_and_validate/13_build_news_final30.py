"""
To-do #7: Build the final merged news table (FinBERT input for Phase 2).

Takes the aligned table from #6, runs integrity checks, adds a stable news_id (hash of
ticker+source+published_utc+text, so IDs survive re-runs and re-sorts), and writes the
final schema.

Input : data/news/news_aligned_final30.csv
Output: data/news/news_final30.csv   (+ news_final30.parquet if pyarrow is installed)
  columns: news_id, feature_date, ticker, source, text_type, text, published_utc,
           outlet, url, match_tier, assignment_rule

Usage (project root):
  python fetch_and_validate/13_build_news_final30.py
"""
import argparse
from pathlib import Path

import pandas as pd
from pandas.util import hash_pandas_object

FINAL_COLS = ["news_id", "feature_date", "ticker", "source", "text_type", "text",
              "published_utc", "outlet", "url", "match_tier", "assignment_rule"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/news/news_aligned_final30.csv")
    ap.add_argument("--out", default="data/news/news_final30.csv")
    ap.add_argument("--tickers", default="data/universe/final_30_tickers.csv")
    ap.add_argument("--prices", default="data/prices/ohlcv_daily.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["published_utc", "feature_date"])
    n_in = len(df)
    final30 = set(pd.read_csv(args.tickers)["ticker"].str.upper().str.strip())
    cal = set(pd.to_datetime(pd.read_csv(args.prices, usecols=["date"])["date"]).dt.normalize())

    problems = []
    if not set(df["ticker"]) <= final30:
        problems.append(f"tickers outside final 30: {sorted(set(df['ticker']) - final30)}")
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        problems.append("null/empty text rows present")
    if not set(df["feature_date"].dt.normalize()) <= cal:
        problems.append("feature_date values that are not trading days")
    if df["feature_date"].isna().any():
        problems.append("null feature_date")
    if problems:
        raise SystemExit("INTEGRITY CHECK FAILED:\n  " + "\n  ".join(problems))

    key = (df["ticker"].astype(str) + "|" + df["source"].astype(str) + "|" +
           df["published_utc"].astype(str) + "|" + df["text"].astype(str))
    df["news_id"] = hash_pandas_object(key, index=False).astype("uint64").map(lambda v: f"{v:016x}")
    dup = int(df["news_id"].duplicated().sum())
    if dup:
        print(f"Dropping {dup} exact duplicate rows (same ticker/source/time/text)")
        df = df.drop_duplicates("news_id")

    df["feature_date"] = df["feature_date"].dt.date
    df = df.sort_values(["feature_date", "ticker", "published_utc", "source"]).reset_index(drop=True)
    df = df[FINAL_COLS]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    parquet_note = ""
    try:
        df.to_parquet(Path(args.out).with_suffix(".parquet"), index=False)
        parquet_note = " + parquet"
    except Exception as e:
        parquet_note = f" (parquet skipped: {type(e).__name__})"

    print(f"Rows in: {n_in:,} | written: {len(df):,} -> {args.out}{parquet_note}")
    print(f"Unique news_id: {df['news_id'].is_unique} | tickers: {df['ticker'].nunique()}/30")
    print(f"feature_date range: {df['feature_date'].min()} -> {df['feature_date'].max()}")

    fd = pd.to_datetime(df["feature_date"])
    print("\nRows per year x source:")
    print(df.assign(year=fd.dt.year).groupby(["year", "source"]).size().unstack(fill_value=0).to_string())

    L = df["text"].str.len()
    print("\nText length in characters (FinBERT truncates at 512 tokens, roughly 2,000 chars):")
    print(df.assign(n=L).groupby("source")["n"].describe(percentiles=[0.5, 0.95, 0.99])[["count", "50%", "95%", "99%", "max"]].round(0).to_string())
    print(f"Rows over 1,500 chars: {(L > 1500).sum():,}")

    print("\nRows per ticker (min / median / max):")
    c = df["ticker"].value_counts()
    print(f"  {c.min():,} ({c.idxmin()}) / {int(c.median()):,} / {c.max():,} ({c.idxmax()})")


if __name__ == "__main__":
    main()
