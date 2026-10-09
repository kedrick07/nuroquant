"""
Rebuild market_cap_rank in candidates.csv.

Fixes the "all None" bug: instead of looping tickers one-by-one (which trips
Yahoo's rate limit after ~30-50 sequential calls and silently returns empty data
for the rest), this does ONE batched price download for all tickers, then fetches
shares_outstanding per ticker with a small delay + retry so it doesn't get throttled.

avg_market_cap_2019_2023 = mean(daily Close, 2019-01-01..2023-12-31) * current shares_outstanding
(This is a proxy for RANKING purposes only, not a precise historical market cap.)

Usage:
  pip install yfinance pandas
  python 0_rank_by_marketcap.py --candidates data/universe/candidates.csv
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_shares_outstanding(ticker: str, retries: int = 2, delay: float = 1.0):
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            shares = t.fast_info.get("shares")
            if shares:
                return shares
            info = t.info
            shares = info.get("sharesOutstanding")
            if shares:
                return shares
        except Exception:
            pass
        time.sleep(delay)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/universe/candidates.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--delay", type=float, default=0.8, help="seconds between shares-outstanding lookups")
    args = ap.parse_args()

    df = pd.read_csv(args.candidates)
    tickers = df["ticker"].str.replace(".", "-", regex=False).tolist()  # yfinance wants BRK-B not BRK.B
    ticker_map = dict(zip(tickers, df["ticker"]))  # back-map to original format for writing results

    print(f"Batch-downloading price history for {len(tickers)} tickers ...")
    prices = yf.download(
        tickers, start=args.start, end=args.end, group_by="ticker",
        auto_adjust=True, threads=True, progress=True,
    )

    avg_close = {}
    for t in tickers:
        try:
            col = prices[t]["Close"] if len(tickers) > 1 else prices["Close"]
            valid = col.dropna()
            avg_close[t] = valid.mean() if len(valid) > 0 else None
        except Exception:
            avg_close[t] = None

    print("Fetching shares outstanding (this part is sequential, be patient) ...")
    shares_out = {}
    for i, t in enumerate(tickers):
        shares_out[t] = fetch_shares_outstanding(t, delay=args.delay)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tickers)} done...")
        time.sleep(args.delay)

    results = []
    for t in tickers:
        orig_ticker = ticker_map[t]
        ac = avg_close.get(t)
        so = shares_out.get(t)
        mcap = ac * so if (ac is not None and so is not None) else None
        results.append({"ticker": orig_ticker, "avg_close_2019_2023": ac, "shares_outstanding": so, "avg_market_cap_2019_2023": mcap})

    res_df = pd.DataFrame(results)
    df = df.drop(columns=["market_cap_rank", "avg_market_cap_2019_2023"], errors="ignore")
    df = df.merge(res_df, on="ticker", how="left")

    n_missing = df["avg_market_cap_2019_2023"].isna().sum()
    df = df.sort_values("avg_market_cap_2019_2023", ascending=False, na_position="last").reset_index(drop=True)
    df["market_cap_rank"] = range(1, len(df) + 1)

    out_path = Path(args.candidates)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} tickers, {n_missing} still missing market cap)")
    if n_missing:
        print("Tickers with missing market cap (fill in manually or re-run for just these):")
        print(df[df["avg_market_cap_2019_2023"].isna()]["ticker"].tolist())
    print(df.head(15)[["market_cap_rank", "ticker", "company_name", "avg_market_cap_2019_2023"]].to_string(index=False))


if __name__ == "__main__":
    main()
