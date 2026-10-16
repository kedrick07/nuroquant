import json
import numpy as np, pandas as pd
from config import *

MACRO_FEATS = ["d_vix_5", "d_dgs10_5", "slope", "d_slope_5", "d_baa10y_20",
               "wti_chg_20", "usd_ret_20", "d_cpi_21", "d_unrate_21", "d_fedfunds_21"]
INTERACTIONS = [(m, s) for m in ["d_vix_5", "d_dgs10_5", "d_baa10y_20", "wti_chg_20"]
                for s in ["beta60", "vol20"]]

def make_macro(m, cal):
    x = m.reindex(cal)
    assert x.notna().all().all(), x.isna().sum()
    f = pd.DataFrame(index=cal)
    f["d_vix_5"] = x["vixcls"].diff(5)
    f["d_dgs10_5"] = x["dgs10"].diff(5)
    slope = x["dgs10"] - x["dgs2"]
    f["slope"] = slope
    f["d_slope_5"] = slope.diff(5)
    f["d_baa10y_20"] = x["baa10y"].diff(20)
    f["wti_chg_20"] = x["dcoilwtico"].diff(20)
    f["usd_ret_20"] = x["dtwexbgs"].pct_change(20)
    for c, k in [("cpi_yoy", "d_cpi_21"), ("unrate", "d_unrate_21"), ("fedfunds", "d_fedfunds_21")]:
        f[k] = x[c].diff(21)
    return f

if __name__ == "__main__":
    m = pd.read_csv(MACRO_FULL)
    m = m.rename(columns={m.columns[0]: "date"})
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date").sort_index()
    m.columns = [c.lower() for c in m.columns]
    need = ["vixcls", "dgs10", "dgs2", "baa10y", "dtwexbgs", "dcoilwtico", "cpi_yoy", "unrate", "fedfunds"]
    assert set(need) <= set(m.columns), m.columns.tolist()

    u7, u8 = m.loc["2020-05-07", "unrate"], m.loc["2020-05-08", "unrate"]
    assert abs(u7 - 4.4) < 0.05 and abs(u8 - 14.8) < 0.05, (u7, u8)
    print("UNRATE release-lag check passed:", u7, u8)

    w = pd.read_csv(OHLCV_WARMUP, usecols=["date"], parse_dates=["date"])["date"]
    cal = pd.DatetimeIndex(sorted(set(w) | set(trading_dates())))
    f = make_macro(m, cal)
    f = f[f.index >= FIRST_LSTM_INPUT_DATE]
    assert f.notna().all().all()
    f0d = pd.DatetimeIndex(sorted(pd.read_parquet(FEATURES_OUT / "f0_ranked.parquet", columns=["date"])["date"].unique()))
    assert f.index.equals(f0d), "macro dates differ from F0 dates"

    rng = np.random.default_rng(2)
    for T in pd.to_datetime(rng.choice(f.index, 6, replace=False)):
        fT = make_macro(m[m.index <= T], cal[cal <= T]).loc[T]
        assert np.allclose(fT.values, f.loc[T].values, equal_nan=True), f"look-ahead at {T}"
    print("look-ahead test passed (6 random dates)")

    out = f.reset_index().rename(columns={"index": "date"})
    out.to_parquet(FEATURES_OUT / "f2_macro_raw.parquet", index=False)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "macro_recipe.json").write_text(json.dumps({
        "macro_feats": MACRO_FEATS, "interactions": INTERACTIONS,
        "stock_side": "rank-normalized beta60, vol20 from f0_ranked",
        "scaling": "fit z-score on training fold only; clip |z| at 5; build products after scaling"}, indent=2))
    print(f.describe().loc[["min", "max"]].round(3).T)
    print("wti_chg_20 extremes:", f["wti_chg_20"].idxmin().date(), f["wti_chg_20"].idxmax().date())