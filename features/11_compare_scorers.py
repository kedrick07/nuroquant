import os
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertTokenizerFast, pipeline
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
lab = pd.read_csv(ROOT / "results/labels/headline_sample.csv", dtype=str, keep_default_na=False)
L = ["positive", "negative", "neutral"]
y = lab["label"].values
ordinal = lab["label"].map({"negative": -1, "neutral": 0, "positive": 1}).values

def report(name, probs):
    pred = probs[L].idxmax(axis=1).values
    f1 = f1_score(y, pred, labels=L, average="macro", zero_division=0)
    rho = spearmanr(probs["positive"] - probs["negative"], ordinal).statistic
    print(f"{name}: macro F1 {f1:.3f}, accuracy {(pred == y).mean():.3f}, "
          f"Spearman(score, label) {rho:.3f}")

def make_pipe(name, sha):
    kw = dict(revision=sha, token=TOKEN)
    opts = dict(top_k=None, device=0, truncation=True, max_length=160)
    try:
        return pipeline("text-classification", model=name, **opts, **kw)
    except ValueError:
        tok = BertTokenizerFast.from_pretrained(name, **kw)
        mdl = BertForSequenceClassification.from_pretrained(name, **kw)
        return pipeline("text-classification", model=mdl, tokenizer=tok, **opts)

sc = pd.read_parquet(ROOT / "data/features/finbert_scores.parquet")
m = lab[["news_id"]].merge(sc, on="news_id", how="left", validate="1:1")
report("ProsusAI/finbert (current)",
       pd.DataFrame({"positive": m["p_pos"], "negative": m["p_neg"], "neutral": m["p_neu"]}))

texts = lab["text"].tolist()
for name in ["yiyanghkust/finbert-tone",
             "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"]:
    sha = HfApi().model_info(name, token=TOKEN).sha
    pipe = make_pipe(name, sha)
    rows = [{d["label"].lower(): d["score"] for d in out} for out in pipe(texts, batch_size=32)]
    report(f"{name} @ {sha[:8]}", pd.DataFrame(rows)[L])