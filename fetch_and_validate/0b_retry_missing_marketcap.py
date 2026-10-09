"""
Retry market cap for tickers that came back None from 0_rank_by_marketcap.py.

Uses get_shares_full() (historical chart endpoint) instead of fast_info/.info
(live quote endpoint) -- this works for tickers that were delisted/renamed/
acquired AFTER the study window, since it doesn't need a currently-active quote.

Computes avg_market_cap_2019_2023 = mean(daily Close * shares_outstanding_as_of_that_date)
over the study window -- more accurate than the static "current shares" proxy too,
so it's fine to also re-run this for ALL tickers if you want the improved calc.

For the handful where even get_shares_full fails (fully deregistered, no data at
all), fill data/universe/manual_market_cap_overrides.csv by hand:
  ticker,avg_market_cap_2019_2023
  WBA,35000000000

Usage:
  python 0b_retry_missing_marketcap.py --candidates data/universe/candidates.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

OVERRIDES_PATH = Path("data/universe/manual_market_cap_overrides.csv")


def compute_market_cap(ticker: str, start: str, end: str):
    yf_ticker = ticker.replace(".", "-")
    t = yf.Ticker(yf_ticker)

    hist = t.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return None

    try:
        shares = t.get_shares_full(start=start, end=end)
    except Exception:
        shares = None

    if shares is None or len(shares) == 0:
        return None

    shares = shares.tz_localize(None) if shares.index.tz is not None else shares
    close = hist["Close"]
    close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index

    shares_daily = shares.reindex(close.index, method="ffill").bfill()
    market_cap = (close * shares_daily).dropna()
    if market_cap.empty:
        return None
    return market_cap.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/universe/candidates.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    df = pd.read_csv(args.candidates)
    missing = df[df["avg_market_cap_2019_2023"].isna()]
    print(f"Retrying {len(missing)} tickers with get_shares_full()...")

    fixed, still_missing = 0, []
    for idx, row in missing.iterrows():
        ticker = row["ticker"]
        mcap = compute_market_cap(ticker, args.start, args.end)
        if mcap is not None:
            df.loc[idx, "avg_market_cap_2019_2023"] = mcap
            fixed += 1
            print(f"  {ticker}: fixed -> {mcap:,.0f}")
        else:
            still_missing.append(ticker)
            print(f"  {ticker}: still no data")

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

    print(f"\nFixed via get_shares_full: {fixed}")
    print(f"Still missing: {len(still_missing)} -> {still_missing}")
    if still_missing:
        print(f"\nAdd these to {OVERRIDES_PATH} manually (ticker,avg_market_cap_2019_2023) and re-run.")


if __name__ == "__main__":
    main()
