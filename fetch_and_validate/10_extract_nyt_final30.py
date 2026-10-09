"""
To-do #4 (v2): Extract NYT Archive articles for the final tickers (precision-first).

v2 changes: --tickers restricts matching to the CURRENT final list (so dropped tickers leave no
stray rows), and patterns were added for CMCSA, TXN, PYPL, WFC, QCOM.

Rules: word-boundary, case-sensitive curated patterns. An article counts only if the company appears
in the HEADLINE, the ABSTRACT/SNIPPET, or the ORGANIZATIONS keyword tag (match_tier headline >
org_keyword > abstract). Text for FinBERT = headline + abstract (snippet fallback).

Input : data/news/nytarchive/raw/YYYY-MM.json
Output: data/news/nytarchive/processed/nytarchive_final30.csv

Usage (project root):
  python fetch_and_validate/10_extract_nyt_final30.py --tickers data/universe/final_30_tickers_v2.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

PATTERNS = {
    "AAPL": [r"(?<!Big )\bApple\b(?! (?:pie|juice|cider|orchard|tree))", r"\bApple Inc\b"],
    "MSFT": [r"\bMicrosoft\b"],
    "AMZN": [r"\bAmazon\b(?! (?:[Rr]ain ?forest|[Rr]iver|[Bb]asin|[Jj]ungle|[Rr]egion|[Ff]ires?|[Dd]eforestation))",
             r"\bAmazon\.com\b"],
    "GOOG": [r"\bGoogle\b", r"\bAlphabet\b(?! soup)"],
    "META": [r"\bFacebook\b", r"\bMeta Platforms\b", r"\bMeta\b(?![-\w])"],
    "NVDA": [r"\bNvidia\b"],
    "V":    [r"\bVisa Inc\b", r"\bVisa and Mastercard\b", r"\bMastercard and Visa\b", r"\bVisa's (?:card|payment|network)"],
    "WMT":  [r"\bWalmart\b", r"\bWal-Mart\b"],
    "JPM":  [r"\bJPMorgan\b", r"\bJ\.P\. Morgan\b", r"\bJP Morgan\b"],
    "XOM":  [r"\bExxon\b", r"\bExxonMobil\b", r"\bExxon Mobil\b"],
    "INTC": [r"\bIntel\b(?! (?:agency|agencies|community))"],
    "DIS":  [r"\bWalt Disney\b", r"\bDisney\b"],
    "ORCL": [r"\bOracle (?:Corp\b|Corporation|Inc\b)", r"\bOracle's (?:cloud|database|software)"],
    "KO":   [r"\bCoca-Cola\b"],
    "CVX":  [r"\bChevron\b"],
    "MRK":  [r"\bMerck\b(?! KGaA)"],
    "PEP":  [r"\bPepsiCo\b", r"\bPepsi\b"],
    "COST": [r"\bCostco\b"],
    "ABBV": [r"\bAbbVie\b"],
    "ADBE": [r"\bAdobe\b(?! (?:brick|house|walls?|home))"],
    "CSCO": [r"\bCisco\b"],
    "NFLX": [r"\bNetflix\b"],
    "TMO":  [r"\bThermo Fisher\b"],
    "ABT":  [r"\bAbbott Laboratories\b", r"\bAbbott Labs\b"],
    "CRM":  [r"\bSalesforce\b"],
    "NKE":  [r"\bNike\b"],
    "ACN":  [r"\bAccenture\b"],
    "DHR":  [r"\bDanaher\b"],
    "AMD":  [r"\bAdvanced Micro Devices\b"],
    "NEE":  [r"\bNextEra\b"],
    "CMCSA": [r"\bComcast\b"],
    "TXN":  [r"\bTexas Instruments\b"],
    "PYPL": [r"\bPayPal\b"],
    "WFC":  [r"\bWells Fargo\b(?! (?:Center|Arena|Field))"],
    "QCOM": [r"\bQualcomm\b"],
}


def compile_patterns(patterns):
    per_ticker = {t: re.compile("|".join(f"(?:{p})" for p in pats)) for t, pats in patterns.items()}
    prefilter = re.compile("|".join(f"(?:{p})" for pats in patterns.values() for p in pats))
    return per_ticker, prefilter


def get_fields(doc):
    headline = ((doc.get("headline") or {}).get("main") or "").strip()
    abstract = (doc.get("abstract") or "").strip()
    snippet = (doc.get("snippet") or "").strip()
    orgs = " | ".join(k.get("value", "") for k in (doc.get("keywords") or [])
                      if k.get("name") == "organizations" and k.get("value"))
    return headline, abstract, snippet, orgs


def match_doc(doc, per_ticker, prefilter):
    headline, abstract, snippet, orgs = get_fields(doc)
    blob = " \n ".join([headline, abstract, snippet, orgs])
    if not prefilter.search(blob):
        return []
    hits = []
    for t, rx in per_ticker.items():
        if rx.search(headline):
            tier = "headline"
        elif rx.search(orgs):
            tier = "org_keyword"
        elif rx.search(abstract) or rx.search(snippet):
            tier = "abstract"
        else:
            continue
        hits.append((t, tier))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/news/nytarchive/raw")
    ap.add_argument("--out", default="data/news/nytarchive/processed/nytarchive_final30.csv")
    ap.add_argument("--tickers", default="data/universe/final_30_tickers_v2.csv")
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2023-12")
    args = ap.parse_args()

    final = set(pd.read_csv(args.tickers)["ticker"].str.upper().str.strip())
    active = {t: p for t, p in PATTERNS.items() if t in final}
    no_pattern = sorted(final - set(PATTERNS))
    if no_pattern:
        print(f"WARNING: no NYT pattern defined for {no_pattern} -> these get no NYT rows")
    print(f"Matching {len(active)} tickers: {sorted(active)}")
    per_ticker, prefilter = compile_patterns(active)

    rows, scanned, missing_months = [], 0, []
    for p in pd.period_range(args.start, args.end, freq="M"):
        fp = Path(args.raw_dir) / f"{p.year}-{p.month:02d}.json"
        if not fp.exists():
            missing_months.append(str(p))
            continue
        with open(fp) as f:
            docs = json.load(f).get("response", {}).get("docs", [])
        scanned += len(docs)
        for d in docs:
            for t, tier in match_doc(d, per_ticker, prefilter):
                headline, abstract, snippet, _ = get_fields(d)
                body = abstract or snippet
                rows.append({
                    "published_utc": d.get("pub_date"), "ticker": t, "headline": headline, "abstract": body,
                    "text": (headline + ". " + body) if body else headline, "match_tier": tier,
                    "section": d.get("section_name"), "type_of_material": d.get("type_of_material"),
                    "url": d.get("web_url"), "doc_id": d.get("_id"),
                })
    print(f"Scanned {scanned:,} docs over {60 - len(missing_months)} months")

    if not rows:
        raise SystemExit("No NYT matches found. Check the raw JSON folder.")
    df = pd.DataFrame(rows)
    df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    bad = int(df["published_utc"].isna().sum())
    df = df.dropna(subset=["published_utc"]).sort_values(["ticker", "published_utc"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nMatched rows written: {len(df):,} -> {args.out}")
    print(f"Bad timestamps dropped: {bad} | months with no raw file: {missing_months or 'none'}")
    print(f"Tickers with NYT matches: {df['ticker'].nunique()}/{len(active)} | none: {sorted(set(active) - set(df['ticker']))}")
    print("\nMatches per ticker / tier:")
    print(df.pivot_table(index="ticker", columns="match_tier", values="doc_id", aggfunc="count", fill_value=0)
            .assign(total=lambda x: x.sum(axis=1)).sort_values("total", ascending=False).to_string())
    print("\nSample headlines per NEW ticker (eyeball for false positives):")
    for t in ["CMCSA", "TXN", "PYPL", "WFC", "QCOM"]:
        g = df[df["ticker"] == t]
        for h in g.sample(min(3, len(g)), random_state=0)["headline"] if len(g) else []:
            print(f"  {t}: {h[:100]}")


if __name__ == "__main__":
    main()