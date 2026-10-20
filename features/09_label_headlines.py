import sys
from pathlib import Path
import pandas as pd

DEFAULT = Path(__file__).resolve().parents[1] / "results" / "labels" / "headline_sample.csv"
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
LAB = {"p": "positive", "n": "negative", "u": "neutral"}
COLS = ["sample_id", "news_id", "ticker", "source", "text", "label", "relevant"]

df = pd.read_csv(PATH, dtype=str, keep_default_na=False)
for c in ("label", "relevant"):
    if c not in df.columns:
        df[c] = ""
df = df[COLS]

def save():
    tmp = PATH.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(PATH)

def is_done(k):
    return df.at[k, "label"] != "" and df.at[k, "relevant"] != ""

i = next((k for k in range(len(df)) if not is_done(k)), len(df))
print("Type two characters per headline:")
print("  1st = p (positive) / n (negative) / u (neutral), from a shareholder's view of THAT ticker")
print("  2nd = 1 (headline is about this company) / 0 (market wrap-up, ETF list, other company)")
print("  examples: p1  n1  u1  u0      b = go back one      q = quit (progress is saved)")
print("Rule: relevant = 0 is always neutral. Use the headline text only; no web searches.")

while i < len(df):
    r = df.loc[i]
    print(f"\n[{i + 1}/{len(df)}] {r['ticker']} | {r['source']}")
    print(r["text"])
    try:
        s = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        break
    if s == "q":
        break
    if s == "b":
        if i > 0:
            i -= 1
            df.at[i, "label"] = ""
            df.at[i, "relevant"] = ""
            save()
        continue
    if len(s) != 2 or s[0] not in LAB or s[1] not in "01":
        print("Use two characters, like p1 or u0 (b = back, q = quit).")
        continue
    label = LAB[s[0]]
    if s[1] == "0" and label != "neutral":
        print("Rule: off-topic -> neutral. Saved as neutral.")
        label = "neutral"
    df.at[i, "label"] = label
    df.at[i, "relevant"] = s[1]
    save()
    i += 1

n_done = sum(is_done(k) for k in range(len(df)))
print(f"\nLabeled {n_done}/{len(df)}.")
if n_done:
    d = df[df["label"] != ""]
    print(d["label"].value_counts().to_dict(), "| relevant=0:", int((d["relevant"] == "0").sum()))