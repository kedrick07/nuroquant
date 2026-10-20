import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
lab = pd.read_csv(ROOT / "results/labels/headline_sample.csv", dtype=str, keep_default_na=False)
sc = pd.read_parquet(ROOT / "data/features/finbert_scores.parquet")
m = lab.merge(sc, on="news_id", validate="1:1")
assert len(m) == 200 and m["label"].isin({"positive", "negative", "neutral"}).all()

names = np.array(["positive", "negative", "neutral"])
L = list(names)
m["pred"] = names[m[["p_pos", "p_neg", "p_neu"]].values.argmax(1)]
y, p = m["label"].values, m["pred"].values

def mf1(a, b):
    return f1_score(a, b, labels=L, average="macro", zero_division=0)

point = mf1(y, p)
rng = np.random.default_rng(42)
boots = np.empty(5000)
for k in range(5000):
    i = rng.integers(0, len(m), len(m))
    boots[k] = mf1(y[i], p[i])
lo, hi = np.percentile(boots, [2.5, 97.5])

print(f"macro F1 {point:.3f} (95% bootstrap CI {lo:.3f} to {hi:.3f}), accuracy {(y == p).mean():.3f}")
for s, g in m.groupby("source"):
    print(f"  {s}: n={len(g)}, macro F1 {mf1(g['label'].values, g['pred'].values):.3f}")
r = m[m["relevant"] == "1"]
print(f"  relevant=1 only: n={len(r)}, macro F1 {mf1(r['label'].values, r['pred'].values):.3f}")
print(classification_report(y, p, labels=L, zero_division=0))
print(pd.DataFrame(confusion_matrix(y, p, labels=L), index=L, columns=L))

if lo > 0.65:
    print("RESULT: pass (CI entirely above 0.65)")
elif hi < 0.65:
    print("RESULT: below target (CI entirely below 0.65): queue a second scorer as a robustness run")
else:
    print("RESULT: CI includes 0.65: apply the pre-declared rule and label 200 more")