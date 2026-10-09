"""
Fetch daily OHLCV price history for the final tickers, 2019-01-01..2023-12-31.

v2: --tickers argument (default data/universe/final_30_tickers_v2.csv).
Single batched yf.download() (avoids the rate-limit cascade) with curl_cffi browser impersonation.
auto_adjust=True gives split/dividend-adjusted prices.

Output: data/prices/ohlcv_daily.csv  (date, ticker, open, high, low, close, volume)

Usage (project root):
  pip install yfinance curl_cffi pandas
  python fetch_and_validate/7_fetch_price_ohlcv.py --tickers data/universe/final_30_tickers_v2.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
    SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    print("curl_cffi not installed -- run: pip install curl_cffi. Falling back to default session.")
    SESSION = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="data/universe/final_30_tickers_v2.csv")
    ap.add_argument("--out", default="data/prices/ohlcv_daily.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    df = pd.read_csv(args.tickers)
    tickers = df["ticker"].str.upper().str.strip().str.replace(".", "-", regex=False).tolist()
    print(f"Fetching daily OHLCV for {len(tickers)} tickers, {args.start} to {args.end} ...")

    raw = yf.download(tickers, start=args.start, end=args.end, group_by="ticker", auto_adjust=True,
                      threads=True, progress=True, session=SESSION)

    frames = []
    for t in tickers:
        try:
            sub = raw[t].copy() if len(tickers) > 1 else raw.copy()
        except KeyError:
            print(f"  {t}: no data returned, skipping")
            continue
        sub = sub.dropna(how="all")
        if sub.empty:
            print(f"  {t}: empty after dropna, skipping")
            continue
        sub = sub.reset_index()
        sub.columns = [c.lower() for c in sub.columns]
        sub["ticker"] = t
        frames.append(sub[["date", "ticker", "open", "high", "low", "close", "volume"]])
        print(f"  {t}: {len(sub)} rows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(args.out, index=False)

    counts = result.groupby("ticker").size()
    print(f"\nWrote {args.out} ({len(result)} rows, {result['ticker'].nunique()} tickers)")
    print(f"Rows per ticker: min {counts.min()} / max {counts.max()}")
    if counts.min() < counts.max():
        print("Warning: uneven row counts across tickers:")
        print(counts.sort_values().head(8).to_string())


if __name__ == "__main__":
    main()