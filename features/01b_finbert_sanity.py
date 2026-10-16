import os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import *
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
rev = (RESULTS / "finbert_revision.txt").read_text().strip()

sc = pd.read_parquet(FEATURES_OUT / "finbert_scores.parquet")
news = pd.read_parquet(NEWS, columns=["news_id", "source", "text"])
assert len(sc) == len(news) and sc["news_id"].is_unique
assert set(sc["news_id"]) == set(news["news_id"])
m = news.merge(sc, on="news_id")
m["cls"] = m[["p_pos", "p_neg", "p_neu"]].idxmax(axis=1)
print(m.groupby("source")["score"].agg(["count", "mean", "std"]).round(3))
print(pd.crosstab(m["source"], m["cls"], normalize="index").round(3))

s = m.sample(500, random_state=0)
tok = AutoTokenizer.from_pretrained(FINBERT_MODEL, revision=rev, token=TOKEN)
mdl = AutoModelForSequenceClassification.from_pretrained(
    FINBERT_MODEL, revision=rev, token=TOKEN).cuda().eval()
lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
cols = [lab["positive"], lab["negative"], lab["neutral"]]
enc = tok(s["text"].tolist(), truncation=True, max_length=FINBERT_MAX_LEN,
          padding=True, return_tensors="pt").to("cuda")
with torch.inference_mode():
    p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()[:, cols]
d = np.abs((p[:, 0] - p[:, 1]) - s["score"].values)
agree = (p.argmax(1) == s[["p_pos", "p_neg", "p_neu"]].values.argmax(1)).mean()
print("fp16 vs fp32: max |score diff|", d.max().round(4),
      "| argmax agreement", round(agree, 4))