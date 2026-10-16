import numpy as np, pandas as pd
from config import *

F0 = ["ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "sma20_ratio", "sma50_ratio",
      "macd_norm", "rsi14", "vol20", "logvol_ratio", "beta60"]
F1 = ["sent_mean", "sent_ewm5", "sent_mom", "abn_count", "days_since_news", "has_news"]
H = 5

p = pd.read_csv(OHLCV, parse_dates=["date"]).sort_values(["ticker", "date"])
px = p.pivot(index="date", columns="ticker", values="close").sort_index()
t = (px.shift(-H) / px - 1).unstack().rename("target_ret5").reset_index()

f0 = pd.read_parquet(FEATURES_OUT / "f0_ranked.parquet")
vol = pd.read_parquet(FEATURES_OUT / "f0_raw.parquet", columns=["date", "ticker", "vol20"]).rename(
    columns={"vol20": "vol20_raw"})
f1 = pd.read_parquet(FEATURES_OUT / "f1_sent_ranked.parquet")
panel = (f0.merge(f1, on=["date", "ticker"], validate="1:1")
           .merge(t, on=["date", "ticker"], how="left", validate="1:1")
           .merge(vol, on=["date", "ticker"], validate="1:1"))
assert len(panel) == len(f0)

g = panel.groupby("date")["target_ret5"]
nvalid = g.transform("count")
valid = nvalid == 30
panel["target_ret5"] = panel["target_ret5"].where(valid)
rk = (panel.groupby("date")["target_ret5"].rank(method="average") - 0.5) / 30 - 0.5
panel["target_rank"] = rk.where(valid)
xm = panel.groupby("date")["target_ret5"].transform("mean")
panel["target_ret5_xs"] = ((panel["target_ret5"] - xm) / (panel["vol20_raw"] * np.sqrt(H))).where(valid)
panel["model_row"] = (panel["date"] >= FIRST_MODEL_DATE) & panel["target_rank"].notna()
panel = panel.drop(columns="vol20_raw").sort_values(["date", "ticker"]).reset_index(drop=True)

assert not panel.duplicated(["date", "ticker"]).any()
assert (panel.groupby("date")["ticker"].nunique() == 30).all()
assert panel[F0 + F1].notna().all().all()
assert panel["date"].min() == FIRST_LSTM_INPUT_DATE

mr = panel[panel["model_row"]]
td = trading_dates()
assert mr["date"].min() == FIRST_MODEL_DATE
assert mr["date"].max() == td[-(H + 1)], (mr["date"].max(), td[-(H + 1)])
assert mr[["target_ret5", "target_rank", "target_ret5_xs"]].notna().all().all()
assert np.allclose(mr.groupby("date")["target_rank"].mean(), 0, atol=1e-9)

rng = np.random.default_rng(3)
for i in rng.choice(len(mr), 6, replace=False):
    r = mr.iloc[i]
    gp = p[p["ticker"] == r["ticker"]].sort_values("date").reset_index(drop=True)
    j = gp.index[gp["date"] == r["date"]][0]
    assert np.isclose(gp.loc[j + H, "close"] / gp.loc[j, "close"] - 1, r["target_ret5"])
print("label alignment test passed (6 random rows)")

macro = pd.read_parquet(FEATURES_OUT / "f2_macro_raw.parquet")
assert set(panel["date"]) <= set(macro["date"])
assert macro.drop(columns="date").notna().all().all()

panel.to_parquet(FEATURES_OUT / "panel_phase2.parquet", index=False)
nd = mr["date"].nunique()
print("model rows", len(mr), "| model dates", nd, "| ~weekly cross-sections", round(nd / H, 1))
print("last model date", mr["date"].max().date(), "| dates per year", mr["date"].dt.year.value_counts().sort_index().to_dict())
print("target_rank min/max", round(mr["target_rank"].min(), 4), round(mr["target_rank"].max(), 4))
print(mr[["target_ret5", "target_ret5_xs"]].describe().loc[["mean", "std", "min", "max"]].round(4))