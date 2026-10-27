# tests/test_cpcv.py
from collections import Counter

import numpy as np
import pytest

from models.cpcv import make_groups, make_combos, split_positions, assign_paths

N_CAL = 1217
MODEL_POS = np.arange(19, 1212)
HORIZON, EMBARGO = 5, 20


@pytest.fixture(scope="module")
def setup():
    return make_groups(MODEL_POS), make_combos()


def all_splits(groups, combos, horizon=HORIZON, embargo=EMBARGO):
    return [split_positions(MODEL_POS, groups, c, N_CAL, horizon, embargo)
            for c in combos]


def has_label_overlap(train, test, horizon=HORIZON):
    test_set = set(test.tolist())
    return any((p + d) in test_set for p in train.tolist() for d in range(horizon + 1))


def has_embargo_breach(train, test, embargo=EMBARGO):
    test_set = set(test.tolist())
    return any((p - d) in test_set for p in train.tolist() for d in range(1, embargo + 1))


def test_groups_partition_model_rows(setup):
    groups, _ = setup
    assert len(groups) == 8
    assert np.array_equal(np.concatenate(groups), MODEL_POS)
    assert sorted(len(g) for g in groups) == [149] * 7 + [150]


def test_combo_counts(setup):
    _, combos = setup
    assert len(combos) == 28
    counts = Counter(g for c in combos for g in c)
    assert all(counts[g] == 7 for g in range(8))


def test_no_label_overlap(setup):
    groups, combos = setup
    for tr, te in all_splits(groups, combos):
        assert not has_label_overlap(tr, te)


def test_embargo_respected(setup):
    groups, combos = setup
    for tr, te in all_splits(groups, combos):
        assert not has_embargo_breach(tr, te)


def test_train_is_valid_model_rows(setup):
    groups, combos = setup
    for tr, te in all_splits(groups, combos):
        assert tr.size > 0 and te.size > 0
        assert np.isin(tr, MODEL_POS).all()
        assert np.intersect1d(tr, te).size == 0


def test_overlap_check_catches_missing_purge(setup):
    groups, combos = setup
    bad = all_splits(groups, combos, horizon=0)
    assert any(has_label_overlap(tr, te) for tr, te in bad)


def test_embargo_check_catches_missing_embargo(setup):
    groups, combos = setup
    bad = all_splits(groups, combos, embargo=0)
    assert any(has_embargo_breach(tr, te) for tr, te in bad)


def test_paths(setup):
    _, combos = setup
    paths = assign_paths(combos)
    assert len(paths) == 7
    assert all(len(p) == 8 for p in paths)
    used = Counter((s, g) for p in paths for g, s in p.items())
    assert len(used) == 56 and max(used.values()) == 1
    assert all(g in combos[s] for p in paths for g, s in p.items())