"""
Fetch NYT Archive API data month-by-month and save raw JSON.

Endpoint: https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key=KEY
Saves each month's raw response to data/news/nytarchive/raw/{year}-{month:02d}.json
Skips months that are already downloaded (resumable).

Requires NYTARCHIVE_API_KEY in a .env file at project root:
  NYTARCHIVE_API_KEY=your_key_here

Usage:
  pip install requests python-dotenv
  python 2_fetch_nytarchive.py --start 2019-01 --end 2023-12
  python 2_fetch_nytarchive.py --start 2019-01 --end 2023-12 --delay 15
"""
import argparse
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

RAW_DIR = Path("data/news/nytarchive/raw")
BASE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"


def fetch_month(year: int, month: int, api_key: str) -> dict:
    url = BASE_URL.format(year=year, month=month)
    resp = requests.get(url, params={"api-key": api_key}, timeout=60)
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429). Increase --delay and retry.")
    resp.raise_for_status()
    return resp.json()


def main():
    load_dotenv()
    api_key = os.getenv("NYTARCHIVE_API_KEY")
    if not api_key:
        raise SystemExit("NYTARCHIVE_API_KEY not found. Add it to your .env file.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    ap.add_argument("--delay", type=float, default=13.0, help="seconds between requests (NYT archive is ~5 req/min)")
    args = ap.parse_args()

    import pandas as pd
    months = pd.period_range(args.start, args.end, freq="M")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetched, skipped, failed = 0, 0, []
    for p in months:
        year, month = p.year, p.month
        out_path = RAW_DIR / f"{year}-{month:02d}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            print(f"Fetching {year}-{month:02d} ...")
            data = fetch_month(year, month, api_key)
            with open(out_path, "w") as f:
                json.dump(data, f)
            n_articles = len(data.get("response", {}).get("docs", []))
            print(f"  saved {n_articles} articles -> {out_path}")
            fetched += 1
        except Exception as e:
            print(f"  FAILED {year}-{month:02d}: {e}")
            failed.append(f"{year}-{month:02d}")
        time.sleep(args.delay)

    print(f"\nDone. fetched={fetched} skipped(existing)={skipped} failed={len(failed)}")
    if failed:
        print("Failed months (re-run script to retry, already-fetched months are skipped):", failed)


if __name__ == "__main__":
    main()
