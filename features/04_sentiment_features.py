import numpy as np, pandas as pd
from config import *

RANK_ALL = ["sent_ewm5", "sent_mom", "abn_count", "days_since_news"]
SENT_FEATS = ["sent_mean", "sent_ewm5", "sent_mom", "abn_count", "days_since_news", "has_news"]
STD_FLOOR, MIN_OBS = 0.10, 5
NVDA_GAP = ["2020-07", "2020-08", "2020-10"] + [f"2020-12", *[f"2021-{m:02d}" for m in range(1, 8)]]

def daily_table(news, scores, dates, tickers):
    n = news.merge(scores[["news_id", "score"]], on="news_id", how="left")
    assert n["score"].notna().all()
    n["feature_date"] = pd.to_datetime(n["feature_date"])
    assert n["feature_date"].isin(dates).all(), "feature_date not a trading day"
    assert n["ticker"].isin(tickers).all()
    g = n.groupby(["feature_date", "ticker"])["score"].agg(base="mean", n="size")
    g.index.names = ["date", "ticker"]
    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    d = g.reindex(grid)
    d["n"] = d["n"].fillna(0)
    return d.reset_index()

def make_feats(d):
    base = d.pivot(index="date", columns="ticker", values="base").sort_index()
    n = d.pivot(index="date", columns="ticker", values="n").sort_index()
    has = (n > 0).astype(float)
    mu = base.rolling(SENT_WINDOW, min_periods=MIN_OBS).mean()
    sd = base.rolling(SENT_WINDOW, min_periods=MIN_OBS).std().clip(lower=STD_FLOOR)
    z = (base - mu) / sd
    undefined = (has > 0) & z.isna()
    zf = z.fillna(0.0)
    ln = np.log1p(n)
    lm = ln.rolling(SENT_WINDOW, min_periods=SENT_MIN_PERIODS).mean()
    ls = ln.rolling(SENT_WINDOW, min_periods=SENT_MIN_PERIODS).std().clip(lower=STD_FLOOR)
    idx = np.arange(len(has))[:, None]
    last = pd.DataFrame(np.where(has.values > 0, idx, np.nan),
                        index=has.index, columns=has.columns).ffill()
    ds = pd.DataFrame(idx - last.values, index=has.index, columns=has.columns).fillna(10).clip(upper=10)
    f = {"sent_mean": zf,
         "sent_ewm5": zf.ewm(halflife=3, adjust=False).mean(),
         "sent_mom": zf.ewm(span=3, adjust=False).mean() - zf.ewm(span=20, adjust=False).mean(),
         "abn_count": ((ln - lm) / ls).fillna(0.0),
         "days_since_news": ds, "has_news": has}
    out = pd.DataFrame({k: v.unstack() for k, v in f.items()})
    out["z_undefined"] = undefined.unstack().astype(int)
    out = out.rename_axis(["ticker", "date"]).reset_index().sort_values(["date", "ticker"])
    return out.reset_index(drop=True)

def rank_sent(df):
    df = df.copy()
    g = df.groupby("date")
    for c in RANK_ALL:
        df[c] = (g[c].rank(method="average") - 0.5) / g[c].transform("count") - 0.5
    m = df["has_news"] == 1
    r = df.loc[m].groupby("date")["sent_mean"]
    df.loc[m, "sent_mean"] = (r.rank(method="average") - 0.5) / r.transform("count") - 0.5
    df.loc[~m, "sent_mean"] = 0.0
    return df

if __name__ == "__main__":
    news = pd.read_parquet(NEWS, columns=["news_id", "feature_date", "ticker"])
    scores = pd.read_parquet(FEATURES_OUT / "finbert_scores.parquet", columns=["news_id", "score"])
    f0 = pd.read_parquet(FEATURES_OUT / "f0_ranked.parquet", columns=["date", "ticker"])
    dates = trading_dates()
    tickers = sorted(f0["ticker"].unique())
    assert len(tickers) == 30

    d = daily_table(news, scores, dates, tickers)
    raw = make_feats(d)
    keep = raw["date"] >= FIRST_LSTM_INPUT_DATE
    raw = raw[keep].reset_index(drop=True)

    key = lambda x: x[["date", "ticker"]].sort_values(["date", "ticker"]).reset_index(drop=True)
    assert key(raw).equals(key(f0)), "F1 rows differ from F0 rows"
    assert raw[SENT_FEATS].notna().all().all()

    rng = np.random.default_rng(1)
    full = raw.set_index(["date", "ticker"])[SENT_FEATS].sort_index()
    for T in pd.to_datetime(rng.choice(raw["date"].unique(), 6, replace=False)):
        dT = daily_table(news[pd.to_datetime(news["feature_date"]) <= T], scores,
                         dates[dates <= T], tickers)
        a = make_feats(dT)
        a = a[a["date"] == T].set_index(["date", "ticker"])[SENT_FEATS].sort_index()
        b = full.loc[[T]].sort_index()
        assert np.allclose(a.values, b.values, equal_nan=True), f"look-ahead at {T}"
    print("look-ahead test passed (6 random dates)")

    dd = d.copy(); dd["month"] = dd["date"].dt.strftime("%Y-%m")
    gap = dd[(dd["ticker"] == "NVDA") & dd["month"].isin(NVDA_GAP)]
    assert gap["n"].sum() == 0, f"NVDA gap months have {gap['n'].sum()} articles"
    print("NVDA gap months: has_news = 0 confirmed")

    try:
        cov = pd.read_csv(NEWS_COVERAGE, parse_dates=["trading_day"])
        cov = cov.rename(columns={"trading_day": "date"})[["date", "ticker", "has_news"]]
        mm = d.merge(cov, on=["date", "ticker"], suffixes=("", "_cov"))
        print("rows compared:", len(mm), "| mismatches:", int(((mm["n"] > 0).astype(int) != mm["has_news"]).sum()))
    except Exception as e:
        print("coverage cross-check skipped:", repr(e))

    ranked = rank_sent(raw)
    assert np.allclose(ranked.groupby("date")[RANK_ALL].mean(), 0, atol=1e-9)
    FEATURES_OUT.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(FEATURES_OUT / "f1_sent_raw.parquet", index=False)
    ranked.drop(columns="z_undefined").to_parquet(FEATURES_OUT / "f1_sent_ranked.parquet", index=False)

    m = ranked["date"] >= FIRST_MODEL_DATE
    print("has_news rate (>= FIRST_MODEL_DATE):", round(ranked.loc[m, "has_news"].mean(), 3))
    print("news-day rows with undefined z (>= FIRST_MODEL_DATE):", int(raw.loc[m, "z_undefined"].sum()))
    print(ranked.loc[m].groupby("ticker")["has_news"].mean().round(2).to_dict())