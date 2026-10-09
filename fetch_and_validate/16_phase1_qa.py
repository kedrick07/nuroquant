"""
To-do #9: Phase 1 QA -- independent end-to-end verification of every Phase 1 artifact.

Checks (each prints PASS/FAIL; exit code 1 if any FAIL):
  universe  : final list size, unique tickers, no duplicate share classes
  prices    : ticker set, rows/ticker, identical date sets, nulls, high>=low, close>0, duplicates
  macro     : columns, continuous daily index, no NaN, plausible ranges, covers every trading day,
              UNRATE release-lag spot check (April-2020 spike must not appear before 2020-05-08)
  news      : unique news_id, tickers, trading-day feature_date, no null text,
              NO-LOOKAHEAD re-check done from raw timestamps (independent of script 12)
  coverage  : daily panel consistent with the news table and with the price calendar;
              only documented exceptions fail the coverage gate
  join test : prices x news-coverage x macro merges with no NaNs

Usage (project root):
  python fetch_and_validate/16_phase1_qa.py
  python fetch_and_validate/16_phase1_qa.py --exceptions NVDA
"""
import argparse
import sys

import numpy as np
import pandas as pd

MACRO_COLS = ["vixcls", "dgs10", "cpi_yoy", "unrate", "dgs2", "baa10y", "dtwexbgs", "dcoilwtico", "fedfunds"]
MACRO_RANGES = {"vixcls": (5, 100), "dgs10": (0, 7), "cpi_yoy": (-2, 12), "unrate": (2, 16), "dgs2": (0, 7),
                "baa10y": (0.5, 6), "dtwexbgs": (90, 140), "dcoilwtico": (-60, 150), "fedfunds": (0, 7)}
DUP_CLASSES = [("GOOG", "GOOGL"), ("NWS", "NWSA"), ("UA", "UAA"), ("FOX", "FOXA"), ("BRK.A", "BRK.B")]

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="data/universe/final_30_tickers_v2.csv")
    ap.add_argument("--prices", default="data/prices/ohlcv_daily.csv")
    ap.add_argument("--macro", default="data/macro/fred_daily.csv")
    ap.add_argument("--news", default="data/news/news_final30.csv")
    ap.add_argument("--daily", default="data/news/news_daily_coverage.csv")
    ap.add_argument("--report", default="data/news/news_coverage_report.csv")
    ap.add_argument("--expected-n", type=int, default=30)
    ap.add_argument("--exceptions", default="NVDA")
    args = ap.parse_args()
    exceptions = {t.strip().upper() for t in args.exceptions.split(",") if t.strip()}

    final = pd.read_csv(args.tickers)["ticker"].str.upper().str.strip().tolist()
    fset = set(final)
    print("=== Universe ===")
    check("final list size", len(final) == args.expected_n, f"{len(final)} tickers")
    check("tickers unique", len(fset) == len(final))
    dups = [p for p in DUP_CLASSES if p[0] in fset and p[1] in fset]
    check("no duplicate share classes", not dups, str(dups))

    print("\n=== Prices ===")
    px = pd.read_csv(args.prices, parse_dates=["date"])
    px["ticker"] = px["ticker"].str.upper()
    check("price tickers == final list", set(px["ticker"]) == fset, f"missing {sorted(fset - set(px['ticker']))}, extra {sorted(set(px['ticker']) - fset)}")
    per = px.groupby("ticker").size()
    check("equal rows per ticker", per.nunique() == 1, f"{per.min()}..{per.max()}")
    ds = px.groupby("ticker")["date"].apply(lambda s: tuple(s.sort_values()))
    check("identical date set for every ticker", ds.nunique() == 1)
    cols = ["open", "high", "low", "close", "volume"]
    check("no nulls in OHLCV", int(px[cols].isna().sum().sum()) == 0)
    check("no duplicate (date,ticker)", int(px.duplicated(["date", "ticker"]).sum()) == 0)
    check("high >= low everywhere", int((px["high"] < px["low"]).sum()) == 0)
    check("close > 0 everywhere", int((px["close"] <= 0).sum()) == 0)
    check("price dates inside 2019-01-01..2023-12-31", px["date"].min() >= pd.Timestamp("2019-01-01") and px["date"].max() <= pd.Timestamp("2023-12-31"),
          f"{px['date'].min().date()} -> {px['date'].max().date()}")
    trading_days = set(px["date"].dt.normalize())
    px = px.sort_values(["ticker", "date"])
    ret = px.groupby("ticker")["close"].pct_change()
    big = px.loc[ret.abs() > 0.25, ["date", "ticker"]].assign(ret=ret[ret.abs() > 0.25].round(3))
    print(f"   info: {len(big)} daily moves > 25% (review): " + "; ".join(f"{r.ticker} {r.date.date()} {r.ret:+.0%}" for r in big.itertuples()))

    print("\n=== Macro ===")
    fred = pd.read_csv(args.macro, parse_dates=["date"])
    check("macro columns", set(fred.columns) == set(["date"] + MACRO_COLS), str(sorted(set(fred.columns) ^ set(["date"] + MACRO_COLS))))
    exp_days = (pd.Timestamp("2023-12-31") - pd.Timestamp("2019-01-01")).days + 1
    check("macro covers every calendar day 2019-2023", len(fred) == exp_days and (fred["date"].diff().dropna() == pd.Timedelta(days=1)).all(), f"{len(fred)} rows, expected {exp_days}")
    check("macro has no NaN", int(fred[[c for c in MACRO_COLS if c in fred]].isna().sum().sum()) == 0)
    bad_rng = [c for c, (lo, hi) in MACRO_RANGES.items() if c in fred and not fred[c].between(lo, hi).all()]
    check("macro values inside plausible ranges", not bad_rng, str(bad_rng))
    check("macro covers every trading day", trading_days <= set(fred["date"].dt.normalize()))
    f2 = fred.set_index("date")
    if pd.Timestamp("2020-05-07") in f2.index and pd.Timestamp("2020-05-08") in f2.index:
        check("UNRATE release lag (Apr-2020 spike not before 2020-05-08)", f2.loc["2020-05-07", "unrate"] < 8 and f2.loc["2020-05-08", "unrate"] > 12,
              f"05-07={f2.loc['2020-05-07','unrate']}, 05-08={f2.loc['2020-05-08','unrate']}")

    print("\n=== News (no-lookahead re-check) ===")
    nw = pd.read_csv(args.news, parse_dates=["published_utc", "feature_date"])
    check("news_id unique", nw["news_id"].is_unique)
    check("news tickers == final list", set(nw["ticker"]) == fset, f"missing {sorted(fset - set(nw['ticker']))}, extra {sorted(set(nw['ticker']) - fset)}")
    check("no null/empty text", int(nw["text"].isna().sum()) == 0 and int((nw["text"].astype(str).str.strip() == "").sum()) == 0)
    check("feature_date is always a trading day", set(nw["feature_date"].dt.normalize()) <= trading_days)
    fn = nw[nw["source"] == "fnspid"]
    ny = nw[nw["source"] == "nyt"]
    v1 = int((fn["feature_date"].dt.normalize() <= fn["published_utc"].dt.normalize()).sum())
    check("FNSPID: feature_date strictly after article date", v1 == 0, f"{v1} violations")
    et = ny["published_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    et_date = et.dt.tz_localize(None).dt.normalize()
    fd = ny["feature_date"].dt.normalize()
    v2 = int((fd < et_date).sum())
    after_close_same_day = (fd == et_date) & (et.dt.hour >= 16)
    v3 = int(after_close_same_day.sum())
    check("NYT: feature_date never before article date (ET)", v2 == 0, f"{v2} violations")
    check("NYT: no same-day assignment after 16:00 ET", v3 == 0, f"{v3} violations")
    if len(ny):
        same = float((fd == et_date).mean() * 100)
        print(f"   info: NYT assigned same trading day: {same:.1f}% (rest pushed: after close, weekend or holiday)")
    print(f"   info: rows {len(nw):,} | FNSPID {len(fn):,} | NYT {len(ny):,} | NYT share {len(ny)/max(len(nw),1)*100:.1f}%")

    print("\n=== Coverage panel ===")
    cov = pd.read_csv(args.daily, parse_dates=["trading_day"])
    check("coverage panel rows == tickers x trading days", len(cov) == len(fset) * len(trading_days), f"{len(cov):,}")
    check("coverage article total == news rows", int(cov["n_articles"].sum()) == len(nw), f"{int(cov['n_articles'].sum()):,} vs {len(nw):,}")
    check("has_news consistent with n_articles", ((cov["n_articles"] > 0).astype(int) == cov["has_news"]).all())
    k1 = set(zip(px["date"], px["ticker"]))
    k2 = set(zip(cov["trading_day"], cov["ticker"]))
    check("coverage keys == price keys", k1 == k2)
    rep = pd.read_csv(args.report)
    fail = set(rep.loc[~rep["passes_gate"].astype(bool), "ticker"])
    check("only documented exceptions fail the coverage gate", fail <= exceptions, f"failing {sorted(fail)}, exceptions {sorted(exceptions)}")
    print(f"   info: {cov['has_news'].mean()*100:.1f}% of ticker-days have news; {(1-cov['has_news'].mean())*100:.1f}% will be neutral-imputed")

    print("\n=== Join test (prices x coverage x macro) ===")
    panel = (px.merge(cov[["trading_day", "ticker", "n_articles", "has_news"]], left_on=["date", "ticker"], right_on=["trading_day", "ticker"], how="left")
               .merge(fred, on="date", how="left"))
    check("join keeps row count", len(panel) == len(px), f"{len(panel):,}")
    check("join has no NaN in news or macro columns", int(panel[["n_articles", "has_news"] + MACRO_COLS].isna().sum().sum()) == 0)

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n=== SUMMARY: {len(checks) - n_fail}/{len(checks)} checks passed ===")
    if n_fail:
        print("FAILED: " + "; ".join(n for n, ok, _ in checks if not ok))
        sys.exit(1)


if __name__ == "__main__":
    main()
