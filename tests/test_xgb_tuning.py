import numpy as np
import pandas as pd
import pytest

from models.tuning import (
    RET, TARGET, XGB_GRID, XGB_SEEDS, select_xgb_config, xgb_complexity,
)

FEATURES = ["x1", "x2"]
SMALL_GRID = [
    {"max_depth": 2, "n_estimators": 20, "min_child_weight": 5},
    {"max_depth": 3, "n_estimators": 40, "min_child_weight": 5},
]
OUTER = np.arange(50, 450)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    calendar = pd.bdate_range("2019-01-01", periods=600)
    n = 30
    df = pd.DataFrame({
        "date": np.repeat(calendar.values, n),
        "ticker": np.tile(np.arange(n), len(calendar)),
    })
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df[RET] = 0.5 * df["x1"] + rng.normal(size=len(df))
    rank = df.groupby("date")[RET].rank()
    df[TARGET] = (rank - 0.5) / n - 0.5
    return df, calendar


def run(df, calendar):
    return select_xgb_config(
        df, calendar, OUTER, FEATURES, calendar[::5],
        grid=SMALL_GRID, seeds=(42,),
    )


def test_returns_grid_member_and_detects_signal(synth):
    df, calendar = synth
    config, mean_ic = run(df, calendar)
    assert config in SMALL_GRID
    assert list(mean_ic.index) == [0, 1]
    assert mean_ic.max() > 0.2


def test_ignores_rows_outside_outer_train(synth):
    df, calendar = synth
    _, ic_a = run(df, calendar)
    df2 = df.copy()
    outside = ~df2["date"].isin(calendar[OUTER])
    df2.loc[outside, [RET, TARGET]] = -df2.loc[outside, [RET, TARGET]]
    _, ic_b = run(df2, calendar)
    pd.testing.assert_series_equal(ic_a, ic_b)


def test_deterministic(synth):
    df, calendar = synth
    _, ic_a = run(df, calendar)
    _, ic_b = run(df, calendar)
    pd.testing.assert_series_equal(ic_a, ic_b)


def test_tie_break_prefers_simpler():
    grid = [
        {"max_depth": 3, "n_estimators": 300, "min_child_weight": 50},
        {"max_depth": 2, "n_estimators": 300, "min_child_weight": 200},
        {"max_depth": 2, "n_estimators": 300, "min_child_weight": 50},
        {"max_depth": 3, "n_estimators": 100, "min_child_weight": 50},
    ]
    assert min(grid, key=xgb_complexity) == grid[3]
    assert min(grid[:3], key=xgb_complexity) == grid[1]


def test_preregistered_grid_and_seeds():
    assert len(XGB_GRID) == 8
    assert len({tuple(sorted(c.items())) for c in XGB_GRID}) == 8
    assert XGB_SEEDS == (42, 43, 44)