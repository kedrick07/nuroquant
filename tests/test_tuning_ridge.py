import numpy as np
import pandas as pd
import pytest

from models.tuning import select_ridge_alpha


def make_panel(n_dates=200, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for d in calendar:
        x = rng.normal(size=(n_tickers, 2))
        y = x[:, 0] + rng.normal(size=n_tickers)
        for t in range(n_tickers):
            rows.append({
                "date": d, "ticker": f"T{t}",
                "f1": x[t, 0], "f2": x[t, 1],
                "target_rank": y[t], "target_ret5": y[t],
            })
    return calendar, pd.DataFrame(rows)


FEATURES = ["f1", "f2"]
GRID = (1.0, 10.0, 1000.0)


def test_returns_grid_value_and_full_table():
    calendar, mr = make_panel()
    best, table = select_ridge_alpha(
        mr, calendar, np.arange(0, 120), FEATURES,
        calendar[::5], grid=GRID,
    )
    assert best in GRID
    assert list(table.index) == list(GRID)
    assert np.isfinite(table.to_numpy()).all()


def test_ignores_rows_outside_outer_train():
    calendar, mr = make_panel()
    outer = np.arange(0, 120)
    poisoned = mr.copy()
    outside = ~poisoned["date"].isin(calendar[outer])
    poisoned.loc[outside, FEATURES + ["target_rank", "target_ret5"]] = np.nan

    best_clean, table_clean = select_ridge_alpha(
        mr, calendar, outer, FEATURES, calendar[::5], grid=GRID,
    )
    best_pois, table_pois = select_ridge_alpha(
        poisoned, calendar, outer, FEATURES, calendar[::5], grid=GRID,
    )
    assert best_clean == best_pois
    pd.testing.assert_series_equal(table_clean, table_pois)


def test_ties_choose_larger_alpha(monkeypatch):
    calendar, mr = make_panel()
    monkeypatch.setattr(
        "models.tuning.rank_ic_by_date",
        lambda df: pd.Series([0.1], index=[0]),
    )
    best, _ = select_ridge_alpha(
        mr, calendar, np.arange(0, 120), FEATURES,
        calendar[::5], grid=GRID,
    )
    assert best == 1000.0


def test_selection_is_deterministic():
    calendar, mr = make_panel()
    a = select_ridge_alpha(
        mr, calendar, np.arange(0, 120), FEATURES,
        calendar[::5], grid=GRID,
    )
    b = select_ridge_alpha(
        mr, calendar, np.arange(0, 120), FEATURES,
        calendar[::5], grid=GRID,
    )
    assert a[0] == b[0]
    pd.testing.assert_series_equal(a[1], b[1])