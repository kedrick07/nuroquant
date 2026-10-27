import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from models.transform import FoldMacroTransformer

MACRO = ["d_vix_5", "d_dgs10_5", "slope", "d_slope_5", "d_baa10y_20",
         "wti_chg_20", "usd_ret_20", "d_cpi_21", "d_unrate_21", "d_fedfunds_21"]
INTER = [[m, s] for m in ["d_vix_5", "d_dgs10_5", "d_baa10y_20", "wti_chg_20"]
         for s in ["beta60", "vol20"]]
RECIPE = {"macro_feats": MACRO, "interactions": INTER}

DATES = pd.bdate_range("2020-01-01", periods=300)
TRAIN, TEST = DATES[:200], DATES[200:]


@pytest.fixture
def macro():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(len(DATES), len(MACRO))),
                        index=DATES, columns=MACRO)


@pytest.fixture
def panel():
    rng = np.random.default_rng(1)
    rows = [(d, f"T{i}") for d in DATES for i in range(5)]
    p = pd.DataFrame(rows, columns=["date", "ticker"])
    p["beta60"] = rng.uniform(-0.5, 0.5, len(p))
    p["vol20"] = rng.uniform(-0.5, 0.5, len(p))
    return p


def test_fit_matches_manual_and_sklearn(macro):
    tf = FoldMacroTransformer(macro, RECIPE).fit(TRAIN)
    manual = macro.loc[TRAIN]
    assert np.allclose(tf.mean_, manual.mean())
    sc = StandardScaler().fit(manual.to_numpy())
    assert np.allclose(tf.mean_.to_numpy(), sc.mean_)
    assert np.allclose(tf.std_.to_numpy(), sc.scale_)


def test_test_dates_do_not_affect_fit(macro):
    macro2 = macro.copy()
    macro2.loc[TEST] = macro2.loc[TEST] + 1000.0
    tf1 = FoldMacroTransformer(macro, RECIPE).fit(TRAIN)
    tf2 = FoldMacroTransformer(macro2, RECIPE).fit(TRAIN)
    pd.testing.assert_series_equal(tf1.mean_, tf2.mean_)
    pd.testing.assert_series_equal(tf1.std_, tf2.std_)
    pd.testing.assert_frame_equal(tf1.transform_macro().loc[TRAIN],
                                  tf2.transform_macro().loc[TRAIN])


def test_shifted_fold_changes_stats(macro):
    a = FoldMacroTransformer(macro, RECIPE).fit(DATES[:200])
    b = FoldMacroTransformer(macro, RECIPE).fit(DATES[100:300])
    assert not np.allclose(a.mean_, b.mean_)
    assert not np.allclose(a.std_, b.std_)


def test_leaky_fit_would_be_caught(macro):
    macro2 = macro.copy()
    macro2.loc[TEST] = macro2.loc[TEST] + 1000.0
    leaky1 = FoldMacroTransformer(macro, RECIPE).fit(DATES)
    leaky2 = FoldMacroTransformer(macro2, RECIPE).fit(DATES)
    assert not np.allclose(leaky1.mean_, leaky2.mean_)


def test_clip_bounds(macro):
    macro = macro.copy()
    macro.loc[TEST[0], "slope"] = 1e6
    z = FoldMacroTransformer(macro, RECIPE).fit(TRAIN).transform_macro()
    assert z.abs().max().max() <= 5.0
    assert z.loc[TEST[0], "slope"] == 5.0


def test_interactions_and_shape(macro, panel):
    out = FoldMacroTransformer(macro, RECIPE).fit(TRAIN).transform(panel)
    assert out.shape == (len(panel), 2 + 10 + 8)
    assert not out.isna().any().any()
    assert out["date"].equals(panel["date"]) and out["ticker"].equals(panel["ticker"])
    for m, s in INTER:
        assert np.allclose(out[f"ix_{m}__{s}"], out[f"z_{m}"] * panel[s])


def test_zero_std_guard(macro):
    macro = macro.copy()
    macro["slope"] = 3.0
    z = FoldMacroTransformer(macro, RECIPE).fit(TRAIN).transform_macro()
    assert np.isfinite(z.to_numpy()).all()
    assert (z["slope"] == 0).all()