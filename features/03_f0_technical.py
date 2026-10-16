import numpy as np, pandas as pd
from config import *

FEATS = ["ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "sma20_ratio", "sma50_ratio",
         "macd_norm", "rsi14", "vol20", "logvol_ratio", "beta60"]

def load_prices():
    a = pd.read_csv(OHLCV_WARMUP, parse_dates=["date"])
    b = pd.read_csv(OHLCV, parse_dates=["date"])
    p = pd.concat([a, b]).sort_values(["ticker", "date"]).reset_index(drop=True)
    assert not p.duplicated(["date", "ticker"]).any()
    return p

def make_raw(p):
    col = "adj_close" if "adj_close" in p.columns else "close"
    px = p.pivot(index="date", columns="ticker", values=col).sort_index()
    vol = p.pivot(index="date", columns="ticker", values="volume").sort_index().replace(0, np.nan)
    r = px.pct_change()
    ew = r.mean(axis=1)
    f = {f"ret_{n}": px.pct_change(n) for n in (1, 5, 10, 20, 60)}
    f["sma20_ratio"] = px / px.rolling(20).mean() - 1
    f["sma50_ratio"] = px / px.rolling(50).mean() - 1
    macd = px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean()
    f["macd_norm"] = macd / px
    d = px.diff()
    ru = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rd = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    f["rsi14"] = 100 - 100 / (1 + ru / rd)
    f["vol20"] = r.rolling(20).std()
    f["logvol_ratio"] = np.log(vol / vol.rolling(20).mean())
    f["beta60"] = r.rolling(60).cov(ew).div(ew.rolling(60).var(), axis=0)
    out = pd.DataFrame({k: v.unstack() for k, v in f.items()})
    out = out.rename_axis(["ticker", "date"]).reset_index().sort_values(["date", "ticker"])
    return out.reset_index(drop=True), col

def rank_norm(df, cols):
    df = df.copy()
    g = df.groupby("date")
    for c in cols:
        df[c] = (g[c].rank(method="average") - 0.5) / g[c].transform("count") - 0.5
    return df

if __name__ == "__main__":
    p = load_prices()
    raw, col = make_raw(p)
    print("price column used:", col)
    raw = raw[raw["date"] >= FIRST_LSTM_INPUT_DATE].reset_index(drop=True)

    assert raw[FEATS].notna().all().all(), raw[FEATS].isna().sum()
    assert (raw.groupby("date")["ticker"].nunique() == 30).all()
    print("rows", len(raw), "| dates", raw["date"].min().date(), "->", raw["date"].max().date())

    full = raw.set_index(["date", "ticker"])
    rng = np.random.default_rng(0)
    for T in pd.to_datetime(rng.choice(raw["date"].unique(), 8, replace=False)):
        trunc, _ = make_raw(p[p["date"] <= T])
        a = trunc[trunc["date"] == T].set_index(["date", "ticker"])[FEATS].sort_index()
        b = full.loc[T][FEATS].sort_index().set_index(a.index)
        assert np.allclose(a.values, b.values, equal_nan=True), f"look-ahead at {T}"
    print("look-ahead test passed (8 random dates)")

    ranked = rank_norm(raw, FEATS)
    assert np.allclose(ranked.groupby("date")[FEATS].mean(), 0, atol=1e-9)
    FEATURES_OUT.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(FEATURES_OUT / "f0_raw.parquet", index=False)
    ranked.to_parquet(FEATURES_OUT / "f0_ranked.parquet", index=False)

    c = ranked[FEATS].corr()
    pairs = [(a, b, round(c.loc[a, b], 3)) for i, a in enumerate(FEATS)
             for b in FEATS[i + 1:] if abs(c.loc[a, b]) > 0.8]
    print("pairs with |corr| > 0.8:", pairs)