import numpy as np
import pytest

from models.tuning import inner_splits


def test_validation_blocks_cover_outer_train_once():
    outer = np.arange(100, 180)
    folds = inner_splits(outer, 300)

    assert len(folds) == 4
    all_valid = np.concatenate([v for _, v in folds])
    np.testing.assert_array_equal(all_valid, outer)


def test_purge_and_embargo_around_every_validation_date():
    outer = np.arange(0, 200)
    for train, valid in inner_splits(outer, 200):
        for v in valid:
            assert not ((train >= v - 5) & (train <= v + 20)).any()
        assert not np.isin(train, valid).any()


def test_gap_in_outer_train_keeps_calendar_positions():
    outer = np.r_[0:40, 100:140]
    folds = inner_splits(outer, 200)

    train1, valid1 = folds[1]
    np.testing.assert_array_equal(valid1, np.arange(20, 40))
    np.testing.assert_array_equal(
        train1, np.r_[0:15, 100:140]
    )

    train2, valid2 = folds[2]
    np.testing.assert_array_equal(valid2, np.arange(100, 120))
    np.testing.assert_array_equal(train2, np.arange(0, 40))


def test_inner_splits_are_deterministic():
    outer = np.arange(50, 250)
    a = inner_splits(outer, 300)
    b = inner_splits(outer, 300)
    for (ta, va), (tb, vb) in zip(a, b):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(va, vb)


@pytest.mark.parametrize(
    "positions, n_calendar, n_splits",
    [
        (np.array([], dtype=int), 100, 4),
        (np.arange(20).reshape(4, 5), 100, 4),
        (np.array([0.0, 1.0, 2.0, 3.0]), 100, 2),
        (np.array([3, 2, 1, 0]), 100, 2),
        (np.array([0, 1, 1, 2]), 100, 2),
        (np.arange(10, 20), 15, 2),
        (np.arange(20), 100, 1),
    ],
)
def test_invalid_inputs_raise(positions, n_calendar, n_splits):
    with pytest.raises(ValueError):
        inner_splits(positions, n_calendar, n_splits=n_splits)