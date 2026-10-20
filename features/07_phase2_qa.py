import json, hashlib, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from config import *

F0 = ["ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "sma20_ratio", "sma50_ratio",
      "macd_norm", "rsi14", "vol20", "logvol_ratio", "beta60"]
F1 = ["sent_mean", "sent_ewm5", "sent_mom", "abn_count", "days_since_news", "has_news"]
RANK_ALL = ["sent_ewm5", "sent_mom", "abn_count", "days_since_news"]
NVDA_GAP = ["2020-07", "2020-08", "2020-10", "2020-12"] + [f"2021-{m:02d}" for m in range(1, 8)]
res = []

def check(name, ok, detail=""):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

panel = pd.read_parquet(FEATURES_OUT / "panel_phase2.parquet")
macro = pd.read_parquet(FEATURES_OUT / "f2_macro_raw.parquet")
td = trading_dates()
tk = pd.read_csv(TICKERS_CSV)
tickers = set(tk["ticker"] if "ticker" in tk.columns else tk.iloc[:, 0])

check("30 tickers match universe file", set(panel["ticker"]) == tickers and len(tickers) == 30)
check("unique (date, ticker) keys", not panel.duplicated(["date", "ticker"]).any())
check("30 stocks every date", (panel.groupby("date")["ticker"].nunique() == 30).all())
exp_cols = set(["date", "ticker"] + F0 + F1 + ["target_ret5", "target_rank", "target_ret5_xs", "model_row"])
check("panel columns exactly as designed (no raw counts)", set(panel.columns) == exp_cols,
      str(set(panel.columns) ^ exp_cols))
check("no NaN in F0/F1 on any row", panel[F0 + F1].notna().all().all())
check("macro covers panel dates, no NaN", set(panel["date"]) <= set(macro["date"]) and macro.drop(columns="date").notna().all().all())
check("macro has WTI dollar change, no WTI return column", "wti_chg_20" in macro.columns and not any("wti_ret" in c for c in macro.columns))

mr = panel[panel["model_row"]]
check("model rows start at FIRST_MODEL_DATE", mr["date"].min() == FIRST_MODEL_DATE)
check("model rows end 5 days before sample end", mr["date"].max() == td[-6])
check("model labels all present", mr[["target_ret5", "target_rank", "target_ret5_xs"]].notna().all().all())
tail = set(panel.loc[(panel["date"] >= FIRST_MODEL_DATE) & ~panel["model_row"], "date"])
check("only last 5 days excluded after FIRST_MODEL_DATE", tail == set(td[-5:]))
check("19 LSTM pre-window dates present", panel.loc[panel["date"] < FIRST_MODEL_DATE, "date"].nunique() == LSTM_LOOKBACK - 1)
check("model dates = 1193", mr["date"].nunique() == 1193, str(mr["date"].nunique()))
check("F0 ranks centered per date", np.allclose(panel.groupby("date")[F0].mean(), 0, atol=1e-9))
check("F1 ranked sentiment centered per date", np.allclose(panel.groupby("date")[RANK_ALL].mean(), 0, atol=1e-9))
check("ranked features within [-0.5, 0.5]", panel[F0 + RANK_ALL + ["sent_mean"]].abs().max().max() <= 0.5 + 1e-9)
check("has_news binary", set(panel["has_news"].unique()) <= {0.0, 1.0})
nv = panel[(panel["ticker"] == "NVDA") & panel["date"].dt.strftime("%Y-%m").isin(NVDA_GAP)]
check("NVDA gap months has_news = 0", len(nv) > 0 and (nv["has_news"] == 0).all() and (nv["sent_mean"] == 0).all())

w = pd.read_csv(OHLCV_WARMUP, parse_dates=["date"])
check("warm-up ends before 2019-01-02, same 30 tickers",
      w["date"].max() < pd.Timestamp("2019-01-02") and set(w["ticker"]) == tickers)
check("no warm-up date in panel", not set(w["date"]) & set(panel["date"]))
check("panel starts at FIRST_LSTM_INPUT_DATE", panel["date"].min() == FIRST_LSTM_INPUT_DATE)

snap = json.loads((RESULTS / "phase2_snapshot.json").read_text())
bad = [k for k, h in snap["files"].items() if sha(ROOT / k) != h]
check("data files match snapshot checksums", not bad, str(bad))
sc = pd.read_parquet(FEATURES_OUT / "finbert_scores.parquet")
news = pd.read_parquet(NEWS, columns=["news_id"])
check("FinBERT cache complete + revision logged",
      len(sc) == len(news) and set(sc["news_id"]) == set(news["news_id"]) and (RESULTS / "finbert_revision.txt").exists()
      and (RESULTS / "finbert_run.json").exists())

lab = pd.read_csv(RESULTS / "labels" / "headline_sample.csv")
lab["label"] = lab["label"].fillna("").astype(str).str.lower().str.strip()
ok_lab = lab["label"].isin({"positive", "negative", "neutral"})
if ok_lab.all():
    m = lab.merge(sc, on="news_id", validate="1:1")
    names = np.array(["positive", "negative", "neutral"])
    m["pred"] = names[m[["p_pos", "p_neg", "p_neu"]].values.argmax(1)]
    L = list(names)
    f1 = f1_score(m["label"], m["pred"], labels=L, average="macro", zero_division=0)
    per = {s: round(f1_score(g["label"], g["pred"], labels=L, average="macro", zero_division=0), 3)
           for s, g in m.groupby("source")}
    acc = (m["label"] == m["pred"]).mean()
    check("FinBERT macro F1 >= 0.65", f1 >= 0.65,
          f"F1 {f1:.3f}, acc {acc:.3f}, by source {per}, kappa {cohen_kappa_score(m['label'], m['pred']):.3f}")
    if "relevant" in m.columns:
        relevant = pd.to_numeric(m["relevant"], errors="coerce")
        if relevant.notna().all() and relevant.isin({0, 1}).all():
            m["relevant"] = relevant
            off_topic_by_source = {
                source: round((group["relevant"] == 0).mean(), 3)
                for source, group in m.groupby("source")
            }
            relevant_rows = m[m["relevant"] == 1]
            relevant_f1 = f1_score(
                relevant_rows["label"], relevant_rows["pred"],
                labels=L, average="macro", zero_division=0
            )
            print(f"Relevant=0 share: {(m['relevant'] == 0).mean():.3f}, by source {off_topic_by_source}")
            print(f"FinBERT macro F1 (relevant=1): {relevant_f1:.3f}")
    print(pd.crosstab(m["label"], m["pred"]))
else:
    check("200 headlines labeled", False, f"{int(ok_lab.sum())}/{len(lab)} valid labels")

print(f"\n{sum(res)}/{len(res)} checks passed")
sys.exit(0 if all(res) else 1)