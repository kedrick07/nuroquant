"""
To-do #3: Extract FNSPID articles for the final 30 tickers.

Looks for the FNSPID CSVs in data/news/fnspid/raw/ first, then data/news/fnspid/.
Streams All_external.csv and nasdaq_exteral_data.csv (chunked), keeps final-30 tickers
within 2019-01-01..2023-12-31, and writes one tidy table.

Decisions applied (see README): FinBERT text for FNSPID = Article_title only.
Lsa_summary is kept in a separate column (summary_extra) for later sensitivity checks.
FB (old Facebook ticker) is mapped to META.

Output: data/news/fnspid/processed/fnspid_final30.csv
  columns: published_utc, ticker, title, publisher, url, summary_extra, source_file

Usage (run from project root):
  python fetch_and_validate/9_extract_fnspid_final30.py
"""
import argparse
from pathlib import Path

import pandas as pd

TICKER_ALIASES = {"FB": "META"}
FILES = ["All_external.csv", "nasdaq_exteral_data.csv"]
WANTED = ["Date", "Stock_symbol", "Article_title", "Url", "Publisher", "Lsa_summary"]
CHUNK = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnspid-dir", default="data/news/fnspid")
    ap.add_argument("--tickers", default="data/universe/final_30_tickers.csv")
    ap.add_argument("--out", default="data/news/fnspid/processed/fnspid_final30.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    final30 = set(pd.read_csv(args.tickers)["ticker"].str.upper().str.strip())
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    fdir = Path(args.fnspid_dir)
    search_dirs = [fdir / "raw", fdir]

    parts = []
    stats = {"scanned": 0, "ticker_match": 0, "bad_date": 0, "out_of_window": 0, "empty_title": 0}
    for fname in FILES:
        path = next((d / fname for d in search_dirs if (d / fname).exists()), None)
        if path is None:
            print(f"[skip] {fname} not found in: {[str(d) for d in search_dirs]}")
            continue
        print(f"Scanning {path} ...")
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in WANTED if c in header]
        for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
            stats["scanned"] += len(chunk)
            sym = chunk["Stock_symbol"].astype(str).str.upper().str.strip().replace(TICKER_ALIASES)
            chunk = chunk.assign(ticker=sym)
            chunk = chunk[chunk["ticker"].isin(final30)]
            stats["ticker_match"] += len(chunk)
            if chunk.empty:
                continue
            ts = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
            stats["bad_date"] += int(ts.isna().sum())
            chunk = chunk.assign(published_utc=ts).dropna(subset=["published_utc"])
            in_win = (chunk["published_utc"] >= start) & (chunk["published_utc"] < end)
            stats["out_of_window"] += int((~in_win).sum())
            chunk = chunk[in_win]
            title = chunk["Article_title"].astype(str).str.strip()
            bad_title = chunk["Article_title"].isna() | (title == "") | (title.str.lower() == "nan")
            stats["empty_title"] += int(bad_title.sum())
            chunk = chunk[~bad_title].assign(title=title[~bad_title])
            out = pd.DataFrame({
                "published_utc": chunk["published_utc"],
                "ticker": chunk["ticker"],
                "title": chunk["title"],
                "publisher": chunk["Publisher"] if "Publisher" in chunk else pd.NA,
                "url": chunk["Url"] if "Url" in chunk else pd.NA,
                "summary_extra": chunk["Lsa_summary"] if "Lsa_summary" in chunk else pd.NA,
                "source_file": fname,
            })
            parts.append(out)

    if not parts:
        raise SystemExit("No FNSPID rows extracted. Check that the CSVs exist in data/news/fnspid/raw/ "
                         "(the second file is spelled nasdaq_exteral_data.csv) and that the ticker file has a 'ticker' column.")

    df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "published_utc"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    midnight = (df["published_utc"].dt.hour.eq(0) & df["published_utc"].dt.minute.eq(0)
                & df["published_utc"].dt.second.eq(0)).mean() * 100
    print(f"\nRows scanned: {stats['scanned']:,}")
    print(f"Final-30 ticker matches: {stats['ticker_match']:,}")
    print(f"Dropped: bad date {stats['bad_date']:,} | outside window {stats['out_of_window']:,} | empty title {stats['empty_title']:,}")
    print(f"Written: {len(df):,} rows -> {args.out}")
    print(f"Tickers present: {df['ticker'].nunique()}/30 | missing: {sorted(final30 - set(df['ticker']))}")
    print(f"Date range: {df['published_utc'].min()} -> {df['published_utc'].max()}")
    print(f"Rows with midnight-only timestamp (time-of-day unknown): {midnight:.1f}%")
    print("\nRows per source file:"); print(df["source_file"].value_counts().to_string())
    print("\nRows per ticker:"); print(df["ticker"].value_counts().to_string())


if __name__ == "__main__":
    main()