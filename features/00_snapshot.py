import hashlib, json, platform, importlib.metadata as md
from config import *

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

files = [OHLCV, OHLCV_WARMUP, MACRO_FULL, NEWS, NEWS_COVERAGE, TICKERS_CSV]
RESULTS.mkdir(exist_ok=True)
out = {"python": platform.python_version(),
       "first_model_date": str(FIRST_MODEL_DATE.date()),
       "first_lstm_input_date": str(FIRST_LSTM_INPUT_DATE.date()),
       "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
       "packages": {k: md.version(k) for k in
                    ["torch", "transformers", "pandas", "numpy",
                     "scikit-learn", "xgboost", "mlflow"] if k in
                    {d.metadata["Name"].lower() for d in md.distributions()}}}
(RESULTS / "phase2_snapshot.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))