"""
6_fetch_macro_fred.py  (v4)

Fetch 9 FRED macro series -> daily panel 2019-01-01..2023-12-31.

Changes vs v3:
  1. CPIAUCSL is converted to YoY % (column renamed cpi_yoy), computed locally from the raw index.
  2. Monthly series (CPI, UNRATE, FEDFUNDS) are shifted to their approximate PUBLIC RELEASE date,
     so a value only appears on the day it could actually have been known (no lookahead).
  3. All series are fetched from 2017-01-01 so YoY and the shifts have history to draw on;
     output is trimmed to the study window.

Release-lag rules (conservative, adjustable in MONTHLY_LAG_DAYS):
  value for reference month M becomes available on (first day of M+1) + lag days
    CPI      : +14  -> ~15th of next month (BLS publishes 2nd/3rd week)
    UNRATE   : +7   -> ~8th of next month  (jobs report = first Friday)
    FEDFUNDS : +1   -> ~2nd of next month  (monthly average, known once the month ends)

Requires FRED_API_KEY in .env
Output: data/macro/fred_daily.csv
"""
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

OUT_DIR = Path("data/macro")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

START = "2019-01-01"
END = "2023-12-31"
FETCH_START = "2017-01-01"

DAILY_SERIES = ["VIXCLS", "DGS10", "DGS2", "BAA10Y", "DTWEXBGS", "DCOILWTICO"]
MONTHLY_LAG_DAYS = {"CPIAUCSL": 14, "UNRATE": 7, "FEDFUNDS": 1}
COLUMN_NAMES = {"CPIAUCSL": "cpi_yoy"}
ORDER = ["VIXCLS", "DGS10", "CPIAUCSL", "UNRATE", "DGS2", "BAA10Y", "DTWEXBGS", "DCOILWTICO", "FEDFUNDS"]


def fetch_series(series_id, api_key, obs_start=FETCH_START, obs_end=END):
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "observation_start": obs_start, "observation_end": obs_end}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "observations" not in data:
        raise RuntimeError(f"{series_id}: unexpected response -> {data}")
    df = pd.DataFrame(data["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].dropna()


def shift_to_release(series, lag_days):
    """Reference-month value -> available on first day of next month + lag_days."""
    out = series.copy()
    out.index = (out.index.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=lag_days)
    return out


def build_panel(raw, start=START, end=END, fetch_start=FETCH_START):
    grid = pd.date_range(fetch_start, end, freq="D")
    panel = pd.DataFrame(index=grid)
    panel.index.name = "date"
    for sid in ORDER:
        s = raw[sid]
        if sid == "CPIAUCSL":
            s = (s.pct_change(12) * 100).dropna()
        if sid in MONTHLY_LAG_DAYS:
            s = shift_to_release(s, MONTHLY_LAG_DAYS[sid])
        s = s[~s.index.duplicated(keep="last")]
        panel[COLUMN_NAMES.get(sid, sid.lower())] = s.reindex(grid).ffill()
    return panel.loc[start:end].reset_index()


def main():
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit("FRED_API_KEY not found. Add it to your .env file.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {}
    for sid in ORDER:
        print(f"Fetching {sid} ...")
        raw[sid] = fetch_series(sid, api_key)
        print(f"  {len(raw[sid])} observations")

    panel = build_panel(raw)
    panel.to_csv(OUT_DIR / "fred_daily.csv", index=False)
    full = build_panel(raw, start="2018-07-01")
    full.to_csv(OUT_DIR / "fred_daily_full.csv", index=False)
    n_nan = int(panel.drop(columns=["date"]).isna().sum().sum())
    print(f"\nWrote {OUT_DIR / 'fred_daily.csv'} ({len(panel)} days x {panel.shape[1]-1} indicators)")
    print(f"Remaining NaN cells: {n_nan}")
    if n_nan:
        print(panel[panel.isna().any(axis=1)].to_string(index=False))


if __name__ == "__main__":
    main()
