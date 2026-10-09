"""
To-do #6: Align cleaned news to trading days (no-lookahead).

feature_date = the first trading day on which an article may be used as a feature.
Prediction is assumed to be made at the market CLOSE of feature_date, so an article must be
public before that close.

Rules
  FNSPID : timestamps are date-only (99.6% midnight), so the true time is unknown.
           Conservative: article dated D -> first trading day strictly AFTER D  (t+1).
  NYT    : real timestamps. Convert UTC -> America/New_York.
           published on a trading day before 16:00 ET -> that same day
           published at/after 16:00 ET, or on a weekend/holiday -> next trading day.
           (--nyt-rule t1 applies the same strict t+1 rule as FNSPID for a uniform lag.)
  Rows whose feature_date would fall after the last trading day in the price file are dropped.

Trading calendar = unique dates in data/prices/ohlcv_daily.csv.

Input : data/news/news_clean_final30.csv, data/prices/ohlcv_daily.csv
Output: data/news/news_aligned_final30.csv  (adds feature_date, assignment_rule, lag_trading_days)

Usage (project root):
  python fetch_and_validate/12_align_news_to_trading_days.py
  python fetch_and_validate/12_align_news_to_trading_days.py --nyt-rule t1
"""
import argparse
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = dtime(16, 0)


def next_trading_day_idx(cal, dates, same_day_ok):
    d64 = dates.values.astype("datetime64[D]")
    left = np.searchsorted(cal, d64, side="left")
    right = np.searchsorted(cal, d64, side="right")
    return np.where(same_day_ok, left, right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="data/news/news_clean_final30.csv")
    ap.add_argument("--prices", default="data/prices/ohlcv_daily.csv")
    ap.add_argument("--out", default="data/news/news_aligned_final30.csv")
    ap.add_argument("--nyt-rule", choices=["exact", "t1"], default="exact")
    args = ap.parse_args()

    news = pd.read_csv(args.news, parse_dates=["published_utc"])
    cal_dates = pd.to_datetime(pd.read_csv(args.prices, usecols=["date"])["date"]).drop_duplicates().sort_values()
    cal = cal_dates.values.astype("datetime64[D]")
    cal_set = set(cal)
    print(f"Trading calendar: {len(cal)} days, {cal_dates.min().date()} -> {cal_dates.max().date()}")

    et = news["published_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    et_date = et.dt.tz_localize(None).dt.normalize()
    pre_close = et.dt.time < CUTOFF
    is_trading = et_date.values.astype("datetime64[D]")
    is_trading = np.array([d in cal_set for d in is_trading])

    is_nyt = news["source"].eq("nyt")
    exact = is_nyt & news["has_time"].astype(bool) & (args.nyt_rule == "exact")

    base_date = pd.Series(np.where(exact, et_date, pd.to_datetime(news["date_utc"])), index=news.index)
    base_date = pd.to_datetime(base_date)
    same_day_ok = exact & pre_close & is_trading

    idx = next_trading_day_idx(cal, base_date, same_day_ok.values)
    in_cal = idx < len(cal)
    feature_date = pd.Series(pd.NaT, index=news.index, dtype="datetime64[ns]")
    feature_date[in_cal] = pd.to_datetime(cal[idx[in_cal]])

    rule = np.where(~is_nyt, "fnspid_date_only_t+1",
           np.where(exact, "nyt_exact_16:00ET", "nyt_t+1"))
    news["feature_date"] = feature_date
    news["assignment_rule"] = rule
    dropped = int((~in_cal).sum())
    news = news[in_cal].copy()
    news["feature_date"] = pd.to_datetime(news["feature_date"]).dt.date

    src_day = pd.to_datetime(np.where(exact[in_cal], et_date[in_cal], pd.to_datetime(news["date_utc"])))
    fd = pd.to_datetime(news["feature_date"])
    assert (fd >= src_day).all(), "feature_date earlier than article date!"
    assert set(fd.values.astype("datetime64[D]")) <= cal_set, "feature_date not a trading day!"
    pos = {d: i for i, d in enumerate(cal)}
    sd = src_day.values.astype("datetime64[D]")
    news["lag_trading_days"] = [pos[f] - int(np.searchsorted(cal, s, side="left")) for f, s in zip(fd.values.astype("datetime64[D]"), sd)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(args.out, index=False)

    print(f"\nRows in: {len(news) + dropped:,} | dropped (feature_date beyond last trading day): {dropped:,} | written: {len(news):,}")
    print("\nRows per assignment rule:"); print(news["assignment_rule"].value_counts().to_string())
    nyt = news[news["source"].eq("nyt") & news["assignment_rule"].eq("nyt_exact_16:00ET")]
    if len(nyt):
        same = (nyt["lag_trading_days"] == 0).mean() * 100
        print(f"\nNYT exact rule: {same:.1f}% same-day, {100 - same:.1f}% pushed to next trading day (after close / weekend / holiday)")
    print("\nTrading-day lag distribution (0 = same day, 1 = next day, ...):")
    print(news.groupby("assignment_rule")["lag_trading_days"].value_counts().sort_index().to_string())
    print(f"\nFeature-date range: {news['feature_date'].min()} -> {news['feature_date'].max()}")
    print(f"Tickers present: {news['ticker'].nunique()}/30 -> {args.out}")


if __name__ == "__main__":
    main()
