import numpy as np
import pandas as pd
import pytest

from models.runner import weekly_eval_dates, rank_ic_by_date


def test_weekly_schedule_uses_model_start():
    calendar = pd.bdate_range("2020-01-01", periods=30)
    model_pos = np.arange(3, 26)

    actual = weekly_eval_dates(calendar, model_pos)

    pd.testing.assert_index_equal(
        actual, calendar[[3, 8, 13, 18, 23]]
    )


def test_weekly_schedule_does_not_restart_in_test_blocks():
    calendar = pd.bdate_range("2020-01-01", periods=30)
    model_pos = np.arange(30)
    test_pos = np.r_[7:13, 22:29]

    global_dates = weekly_eval_dates(calendar, model_pos)
    selected = global_dates[
        global_dates.isin(calendar[test_pos])
    ]

    pd.testing.assert_index_equal(
        selected, calendar[[10, 25]]
    )


@pytest.mark.parametrize(
    "positions",
    [
        np.array([], dtype=int),
        np.array([0, 1, 3]),
        np.array([2, 1, 0]),
        np.array([-1, 0, 1]),
        np.array([28, 29, 30]),
        np.array([0.0, 1.0, 2.0]),
    ],
)
def test_weekly_schedule_rejects_invalid_positions(positions):
    calendar = pd.bdate_range("2020-01-01", periods=30)

    with pytest.raises(ValueError):
        weekly_eval_dates(calendar, positions)


def test_rank_ic_is_computed_separately_per_date():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({
        "date": np.repeat(dates, 4),
        "target_ret5": [1, 2, 3, 4, 1, 2, 3, 4],
        "pred": [1, 2, 3, 4, 4, 3, 2, 1],
    })

    actual = rank_ic_by_date(frame)

    assert actual.loc[dates[0]] == pytest.approx(1.0)
    assert actual.loc[dates[1]] == pytest.approx(-1.0)


def test_rank_ic_handles_ties():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 4),
        "target_ret5": [1, 2, 2, 4],
        "pred": [1, 1, 3, 4],
    })

    expected = frame["pred"].corr(
        frame["target_ret5"], method="spearman"
    )

    assert rank_ic_by_date(frame).iloc[0] == pytest.approx(expected)


def test_rank_ic_preserves_undefined_constant_prediction():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02"] * 4),
        "target_ret5": [1, 2, 3, 4],
        "pred": [0, 0, 0, 0],
    })

    with np.errstate(divide="ignore", invalid="ignore"):
        actual = rank_ic_by_date(frame)

    assert pd.isna(actual.iloc[0])