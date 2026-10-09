"""
Retry v2: fixes the real cause of the 0/26 failures -- Yahoo's bot-protection
blocking yfinance's plain requests session (not real delistings). Confirmed
fix per yfinance maintainers/community: use a curl_cffi session that
impersonates a real browser TLS fingerprint, plus real delay + exponential
backoff between calls (the v1 retry script had ZERO delay, which is why every
single one failed identically).

pip install -U yfinance curl_cffi

Usage:
  python 0c_retry_missing_marketcap_v2.py --candidates data/universe/candidates.csv
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
    SESSION = curl_requests.Session(impersonate="chrome")
except ImportError:
    print("curl_cffi not installed -- run: pip install curl_cffi")
    SESSION = None

OVERRIDES_PATH = Path("data/universe/manual_market_cap_overrides.csv")


def compute_market_cap(ticker: str, start: str, end: str, retries: int = 3, base_delay: float = 4.0):
    yf_ticker = ticker.replace(".", "-")
    for attempt in range(retries):
        try:
            t = yf.Ticker(yf_ticker, session=SESSION)
            hist = t.history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                raise ValueError("empty history")

            shares = t.get_shares_full(start=start, end=end)
            if shares is None or len(shares) == 0:
                raise ValueError("empty shares")

            shares = shares.tz_localize(None) if shares.index.tz is not None else shares
            close = hist["Close"]
            close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
            shares_daily = shares.reindex(close.index, method="ffill").bfill()
            market_cap = (close * shares_daily).dropna()
            if not market_cap.empty:
                return market_cap.mean()
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"    attempt {attempt+1} failed ({e}); retrying in {wait:.0f}s")
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/universe/candidates.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--delay", type=float, default=5.0, help="seconds between tickers")
    args = ap.parse_args()

    df = pd.read_csv(args.candidates)
    missing = df[df["avg_market_cap_2019_2023"].isna()]
    print(f"Retrying {len(missing)} tickers with curl_cffi session + backoff...")

    fixed, still_missing = 0, []
    for idx, row in missing.iterrows():
        ticker = row["ticker"]
        print(f"  {ticker} ...")
        mcap = compute_market_cap(ticker, args.start, args.end)
        if mcap is not None:
            df.loc[idx, "avg_market_cap_2019_2023"] = mcap
            fixed += 1
            print(f"  {ticker}: fixed -> {mcap:,.0f}")
        else:
            still_missing.append(ticker)
            print(f"  {ticker}: still no data after retries")
        time.sleep(args.delay)

    if still_missing and OVERRIDES_PATH.exists():
        overrides = pd.read_csv(OVERRIDES_PATH)
        override_map = dict(zip(overrides["ticker"], overrides["avg_market_cap_2019_2023"]))
        for ticker in list(still_missing):
            if ticker in override_map:
                df.loc[df["ticker"] == ticker, "avg_market_cap_2019_2023"] = override_map[ticker]
                still_missing.remove(ticker)
                print(f"  {ticker}: filled from manual override")

    df = df.sort_values("avg_market_cap_2019_2023", ascending=False, na_position="last").reset_index(drop=True)
    df["market_cap_rank"] = range(1, len(df) + 1)
    df.to_csv(args.candidates, index=False)

    print(f"\nFixed: {fixed}  |  Still missing: {len(still_missing)} -> {still_missing}")
    if still_missing:
        print(f"Add these to {OVERRIDES_PATH} manually (ticker,avg_market_cap_2019_2023) and re-run once more.")


if __name__ == "__main__":
    main()
