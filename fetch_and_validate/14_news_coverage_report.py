"""
To-do #8: News coverage report for the final 30 tickers (re-runs the README coverage gate
on the cleaned data) + the daily has_news panel that Phase 2 uses as the coverage indicator.

Month coverage is counted on feature_date (the trading day an article becomes usable).

Gate (README 2.1): avg >= 5 articles/month AND >= 85% of the 60 months with >= 1 article.
Reliability flag: 100% of months covered.

Inputs : data/news/news_final30.csv, data/prices/ohlcv_daily.csv
Outputs: data/news/news_coverage_report.csv   (one row per ticker)
         data/news/news_daily_coverage.csv    (trading_day x ticker: n_articles, n_fnspid, n_nyt, has_news)

Usage (project root):
  python fetch_and_validate/14_news_coverage_report.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GATE_AVG, GATE_PCT = 5.0, 85.0


def longest_zero_run(flags):
    best = cur = 0
    for v in flags:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="data/news/news_final30.csv")
    ap.add_argument("--prices", default="data/prices/ohlcv_daily.csv")
    ap.add_argument("--report", default="data/news/news_coverage_report.csv")
    ap.add_argument("--daily", default="data/news/news_daily_coverage.csv")
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    args = ap.parse_args()

    news = pd.read_csv(args.news, parse_dates=["feature_date"])
    px = pd.read_csv(args.prices, usecols=["date", "ticker"], parse_dates=["date"])
    cal = np.sort(px["date"].unique())
    tickers = sorted(news["ticker"].unique())
    months = pd.period_range(args.start, args.end, freq="M")
    n_months = len(months)

    news["ym"] = news["feature_date"].dt.to_period("M")
    mcount = news.groupby(["ticker", "ym"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    mcount = mcount.reindex(tickers, fill_value=0)

    daily = (news.groupby(["feature_date", "ticker", "source"]).size().unstack("source", fill_value=0)
             .rename(columns={"fnspid": "n_fnspid", "nyt": "n_nyt"}))
    for c in ("n_fnspid", "n_nyt"):
        if c not in daily:
            daily[c] = 0
    full = pd.MultiIndex.from_product([pd.DatetimeIndex(cal), tickers], names=["trading_day", "ticker"])
    daily = daily.reset_index().rename(columns={"feature_date": "trading_day"}).set_index(["trading_day", "ticker"])
    daily = daily.reindex(full, fill_value=0)
    daily["n_articles"] = daily["n_fnspid"] + daily["n_nyt"]
    daily["has_news"] = (daily["n_articles"] > 0).astype(int)
    daily = daily.reset_index()
    Path(args.daily).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.daily, index=False)

    rows = []
    for t in tickers:
        mc = mcount.loc[t]
        d = daily[daily["ticker"] == t].sort_values("trading_day")
        yr = d.groupby(d["trading_day"].dt.year)["has_news"].mean() * 100
        nyt_share = d["n_nyt"].sum() / max(d["n_articles"].sum(), 1) * 100
        rows.append({
            "ticker": t,
            "total_articles": int(mc.sum()),
            "avg_per_month": round(mc.sum() / n_months, 1),
            "months_covered": int((mc >= 1).sum()),
            "pct_months_covered": round((mc >= 1).mean() * 100, 1),
            "months_under_3": int((mc < 3).sum()),
            "pct_trading_days_with_news": round(d["has_news"].mean() * 100, 1),
            "longest_news_gap_days": longest_zero_run(d["has_news"].values),
            "worst_year_days_pct": round(yr.min(), 1),
            "nyt_share_pct": round(nyt_share, 1),
            "passes_gate": bool(mc.sum() / n_months >= GATE_AVG and (mc >= 1).mean() * 100 >= GATE_PCT),
            "fully_covered_months": bool((mc >= 1).all()),
        })
    rep = pd.DataFrame(rows).sort_values(["pct_months_covered", "avg_per_month"], ascending=[True, True])
    rep.to_csv(args.report, index=False)

    print(f"Calendar: {len(cal)} trading days x {len(tickers)} tickers = {len(daily):,} ticker-days -> {args.daily}")
    print(f"Overall: {daily['has_news'].mean()*100:.1f}% of ticker-days have news; "
          f"{(1-daily['has_news'].mean())*100:.1f}% will be imputed neutral\\n")
    print(rep.to_string(index=False))
    fail = rep[~rep["passes_gate"]]
    notfull = rep[~rep["fully_covered_months"]]
    print(f"\nFail gate: {fail['ticker'].tolist() or 'none'}")
    print(f"Not 100% month coverage: {notfull['ticker'].tolist() or 'none'}")
    print(f"-> {args.report}")


if __name__ == "__main__":
    main()
