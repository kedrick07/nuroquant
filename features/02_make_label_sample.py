import pandas as pd, numpy as np
from config import *

n = pd.read_parquet(NEWS, columns=["news_id", "feature_date", "ticker", "source", "text"])
n["year"] = pd.to_datetime(n["feature_date"]).dt.year
n = n[n["year"].between(2019, 2023)]
is_nyt = n["source"].str.lower().str.contains("nyt|new york")
print(n["source"].value_counts().to_dict())

parts = []
for y, g in n.groupby("year"):
    nyt, fn = g[is_nyt.loc[g.index]], g[~is_nyt.loc[g.index]]
    a = nyt.sample(min(6, len(nyt)), random_state=42)
    w = 1 / fn.groupby("ticker")["ticker"].transform("count")
    b = fn.sample(40 - len(a), weights=w, random_state=42)
    parts += [a, b]

s = pd.concat(parts).drop_duplicates("text").sample(frac=1, random_state=7).reset_index(drop=True)
s.insert(0, "sample_id", range(1, len(s) + 1))
s["label"] = ""
out = RESULTS / "labels"; out.mkdir(parents=True, exist_ok=True)
s[["sample_id", "news_id", "ticker", "source", "text", "label"]].to_csv(
    out / "headline_sample.csv", index=False)
print(len(s), s["source"].value_counts().to_dict(), s["year"].value_counts().sort_index().to_dict())