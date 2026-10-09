"""
To-do #5: Clean, deduplicate and relevance-filter FNSPID + NYT news for the final 30 tickers.

Inputs
  data/news/fnspid/processed/fnspid_final30.csv       (from #3)
  data/news/nytarchive/processed/nytarchive_final30.csv (from #4)

Rules
  FNSPID : drop titles that are empty or very short; dedupe by (ticker,url); dedupe by
           (ticker, calendar day, normalized title)  -> syndicated copies of one headline collapse to one.
  NYT    : 1) keep only articles where the company appears in the HEADLINE or ABSTRACT
              (match_tier headline/abstract). Rows matched only via the organizations tag are dropped:
              FinBERT would score text that never mentions the company.
           2) keep only news-type sections (KEEP_SECTIONS)
           3) drop non-news formats (DROP_TYPES: op-eds, reviews, letters, obits, briefings, ...)
           4) dedupe by (doc_id,ticker) and (ticker, day, normalized headline)

Outputs
  data/news/news_clean_final30.csv            unified long table (FinBERT input)
  data/news/nytarchive/processed/nyt_dropped_orgtag_only.csv   reference copy of the tag-only rows
  data/news/cleaning_report.csv               step-by-step row counts per source

Unified columns: published_utc, date_utc, ticker, source, text_type, text, outlet, url, match_tier, has_time

Usage (project root):
  python fetch_and_validate/11_clean_news_final30.py
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

KEEP_SECTIONS = {"Business Day", "Technology", "U.S.", "World", "Health", "Climate", "Science",
                 "Politics", "New York", "Washington"}
DROP_TYPES = {"Op-Ed", "Editorial", "Letter", "Review", "Obituary (Obit)", "Interactive Feature",
              "Video", "Slideshow", "List", "briefing", "Briefing", "Podcast", "Quote", "Question"}
MIN_TITLE_CHARS = 15


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def has_time(ts):
    return ~(ts.dt.hour.eq(0) & ts.dt.minute.eq(0) & ts.dt.second.eq(0))


def clean_fnspid(path, report):
    df = pd.read_csv(path, parse_dates=["published_utc"])
    n0 = len(df)
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() >= MIN_TITLE_CHARS]
    n1 = len(df)
    df = df.drop_duplicates(["ticker", "url"], keep="first") if df["url"].notna().any() else df
    n2 = len(df)
    df["day"] = df["published_utc"].dt.normalize()
    df["nt"] = df["title"].map(norm)
    df = df.drop_duplicates(["ticker", "day", "nt"], keep="first")
    n3 = len(df)
    for step, n in [("input", n0), ("drop short/empty title", n1), ("dedupe ticker+url", n2), ("dedupe ticker+day+title", n3)]:
        report.append({"source": "fnspid", "step": step, "rows": n})
    return pd.DataFrame({
        "published_utc": df["published_utc"], "date_utc": df["day"].dt.date, "ticker": df["ticker"],
        "source": "fnspid", "text_type": "fnspid_title", "text": df["title"],
        "outlet": df["publisher"], "url": df["url"], "match_tier": "title", "has_time": has_time(df["published_utc"]),
    })


def clean_nyt(path, report, orgtag_out):
    df = pd.read_csv(path, parse_dates=["published_utc"])
    report.append({"source": "nyt", "step": "input", "rows": len(df)})
    tag_only = df[df["match_tier"] == "org_keyword"]
    Path(orgtag_out).parent.mkdir(parents=True, exist_ok=True)
    tag_only.to_csv(orgtag_out, index=False)
    df = df[df["match_tier"].isin(["headline", "abstract"])]
    report.append({"source": "nyt", "step": "keep headline/abstract matches only", "rows": len(df)})
    dropped_sec = df.loc[~df["section"].isin(KEEP_SECTIONS), "section"].value_counts()
    df = df[df["section"].isin(KEEP_SECTIONS)]
    report.append({"source": "nyt", "step": "keep news sections", "rows": len(df)})
    dropped_type = df.loc[df["type_of_material"].isin(DROP_TYPES), "type_of_material"].value_counts()
    df = df[~df["type_of_material"].isin(DROP_TYPES)]
    report.append({"source": "nyt", "step": "drop non-news formats", "rows": len(df)})
    df = df.drop_duplicates(["doc_id", "ticker"])
    df["day"] = df["published_utc"].dt.normalize()
    df["nt"] = df["headline"].map(norm)
    df = df.drop_duplicates(["ticker", "day", "nt"])
    report.append({"source": "nyt", "step": "dedupe", "rows": len(df)})
    out = pd.DataFrame({
        "published_utc": df["published_utc"], "date_utc": df["day"].dt.date, "ticker": df["ticker"],
        "source": "nyt", "text_type": "nyt_headline_abstract", "text": df["text"],
        "outlet": "NYT/" + df["section"].astype(str), "url": df["url"], "match_tier": df["match_tier"],
        "has_time": has_time(df["published_utc"]),
    })
    return out, dropped_sec, dropped_type


def check_raw_dupes(raw_dir, months):
    for m in months:
        fp = Path(raw_dir) / f"{m}.json"
        if not fp.exists():
            continue
        docs = json.load(open(fp)).get("response", {}).get("docs", [])
        ids = pd.Series([d.get("_id") for d in docs])
        print(f"  raw {m}: {len(docs):,} docs, {int(ids.duplicated().sum()):,} duplicate _id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnspid", default="data/news/fnspid/processed/fnspid_final30.csv")
    ap.add_argument("--nyt", default="data/news/nytarchive/processed/nytarchive_final30.csv")
    ap.add_argument("--out", default="data/news/news_clean_final30.csv")
    ap.add_argument("--report", default="data/news/cleaning_report.csv")
    ap.add_argument("--orgtag-out", default="data/news/nytarchive/processed/nyt_dropped_orgtag_only.csv")
    ap.add_argument("--raw-dir", default="data/news/nytarchive/raw")
    args = ap.parse_args()

    report = []
    f = clean_fnspid(args.fnspid, report)
    n, dsec, dtype = clean_nyt(args.nyt, report, args.orgtag_out)
    allnews = pd.concat([f, n], ignore_index=True)

    before = len(allnews)
    allnews["nt"] = allnews["text"].map(norm)
    allnews = allnews.drop_duplicates(["ticker", "date_utc", "nt"]).drop(columns="nt")
    report.append({"source": "both", "step": "cross-source dedupe ticker+day+text", "rows": len(allnews)})
    allnews = allnews.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    allnews.to_csv(args.out, index=False)
    pd.DataFrame(report).to_csv(args.report, index=False)

    print("=== Cleaning funnel ===")
    print(pd.DataFrame(report).to_string(index=False))
    print(f"\nCross-source duplicates removed: {before - len(allnews)}")
    print("\nNYT sections dropped (top 10):"); print(dsec.head(10).to_string())
    print("\nNYT formats dropped:"); print(dtype.to_string())
    print(f"\nFinal table: {len(allnews):,} rows -> {args.out}")
    print(allnews.groupby(["ticker", "source"]).size().unstack(fill_value=0)
          .assign(total=lambda x: x.sum(axis=1)).sort_values("total", ascending=False).to_string())
    print(f"\nTickers present: {allnews['ticker'].nunique()}/30")
    print("\nOdd raw months (duplicate _id check):")
    check_raw_dupes(args.raw_dir, ["2021-01", "2023-03", "2019-01"])


if __name__ == "__main__":
    main()
