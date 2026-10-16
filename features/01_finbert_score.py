import os, json, time
import numpy as np, pandas as pd, torch, transformers
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import *

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
OUT = FEATURES_OUT / "finbert_scores.parquet"
REV_FILE = RESULTS / "finbert_revision.txt"
BATCH = 128
FEATURES_OUT.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True)

if OUT.exists():
    raise SystemExit(f"{OUT} exists, delete it to re-score")

if REV_FILE.exists():
    rev = REV_FILE.read_text().strip()
else:
    rev = HfApi().model_info(FINBERT_MODEL, token=TOKEN).sha
    REV_FILE.write_text(rev)
print("revision:", rev)

news = pd.read_parquet(NEWS, columns=["news_id", "text"])
assert news["news_id"].is_unique
news["text"] = news["text"].fillna("").astype(str).str.strip()
assert (news["text"].str.len() > 0).all(), "empty texts found"

tokz = AutoTokenizer.from_pretrained(FINBERT_MODEL, revision=rev, token=TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(
    FINBERT_MODEL, revision=rev, token=TOKEN).half().cuda().eval()
lab = {v.lower(): k for k, v in model.config.id2label.items()}
ip, ineg, ineu = lab["positive"], lab["negative"], lab["neutral"]

enc = tokz(news["text"].tolist(), truncation=True, max_length=FINBERT_MAX_LEN,
           add_special_tokens=True)["input_ids"]
lens = np.array([len(x) for x in enc])
print("truncated at max_length:", int((lens >= FINBERT_MAX_LEN).sum()))
order = np.argsort(lens)
probs = np.zeros((len(enc), 3), dtype=np.float32)

t0 = time.time()
with torch.inference_mode():
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        b = tokz.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt")
        b = {k: v.cuda() for k, v in b.items()}
        p = torch.softmax(model(**b).logits.float(), dim=-1).cpu().numpy()
        probs[idx] = p[:, [ip, ineg, ineu]]
        if (i // BATCH) % 100 == 0:
            print(f"{i}/{len(order)}  {time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame({"news_id": news["news_id"].values,
                    "p_pos": probs[:, 0], "p_neg": probs[:, 1], "p_neu": probs[:, 2]})
res["score"] = res["p_pos"] - res["p_neg"]
assert res[["p_pos", "p_neg", "p_neu"]].sum(axis=1).sub(1).abs().max() < 1e-2
assert res["score"].notna().all() and len(res) == len(news)
res.to_parquet(OUT, index=False)

argmax = res[["p_pos", "p_neg", "p_neu"]].idxmax(axis=1).value_counts(normalize=True)
print(argmax.round(3).to_dict(), "mean score", round(res["score"].mean(), 4))
(RESULTS / "finbert_run.json").write_text(json.dumps({
    "model": FINBERT_MODEL, "revision": rev, "max_length": FINBERT_MAX_LEN,
    "n_scored": len(res), "seconds": round(time.time() - t0),
    "torch": torch.__version__, "transformers": transformers.__version__}, indent=2))