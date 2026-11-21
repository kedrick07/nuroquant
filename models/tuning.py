"""Inner purged folds for nested model selection."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.cpcv import split_positions
from models.runner import rank_ic_by_date, TARGET, RET

RIDGE_GRID = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)

def select_ridge_alpha(mr, calendar, outer_train_pos, features,
                       weekly_dates, grid=RIDGE_GRID):
    """Pick alpha by mean weekly rank IC over the inner validation folds."""
    folds = inner_splits(outer_train_pos, len(calendar))
    weekly_dates = pd.DatetimeIndex(weekly_dates)
    rows = []

    for k, (tr_pos, va_pos) in enumerate(folds):
        tr = mr[mr["date"].isin(calendar[tr_pos])]
        va = mr[
            mr["date"].isin(calendar[va_pos])
            & mr["date"].isin(weekly_dates)
        ]
        for alpha in grid:
            model = Ridge(alpha=alpha, solver="svd")
            model.fit(tr[features], tr[TARGET])
            scored = va[["date", RET]].copy()
            scored["pred"] = model.predict(va[features])
            rows.append({
                "fold": k,
                "alpha": alpha,
                "ic": rank_ic_by_date(scored).mean(),
            })

    mean_ic = pd.DataFrame(rows).groupby("alpha")["ic"].mean()
    best = mean_ic.index[mean_ic >= mean_ic.max() - 1e-12].max()
    return float(best), mean_ic

def inner_splits(
    outer_train_pos,
    n_calendar,
    n_splits=4,
    horizon=5,
    embargo=20,
):
    """Return inner train/validation pairs as original calendar positions."""
    positions = np.asarray(outer_train_pos)

    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("outer_train_pos must be a non-empty 1D array")
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("outer_train_pos must contain integer positions")
    if not isinstance(n_calendar, (int, np.integer)) or n_calendar <= 0:
        raise ValueError("n_calendar must be a positive integer")
    if not isinstance(n_splits, (int, np.integer)):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2 or n_splits > len(positions):
        raise ValueError("n_splits must be between 2 and the number of dates")
    if positions[0] < 0 or positions[-1] >= n_calendar:
        raise ValueError("positions are outside the calendar")
    if not np.all(np.diff(positions) > 0):
        raise ValueError("positions must be sorted and unique")
    for name, value in (("horizon", horizon), ("embargo", embargo)):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    groups = np.array_split(positions, n_splits)
    folds = []

    for fold_id in range(n_splits):
        train_pos, valid_pos = split_positions(
            positions,
            groups,
            (fold_id,),
            n_calendar,
            horizon=horizon,
            embargo=embargo,
        )
        if len(train_pos) == 0:
            raise ValueError(f"inner fold {fold_id} has no training dates")
        folds.append((train_pos, valid_pos))

    return folds