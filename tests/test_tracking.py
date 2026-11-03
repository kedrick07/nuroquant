import mlflow
import pytest
import numpy as np
from mlflow.tracking import MlflowClient
from models.cpcv import make_groups, make_combos
from models.tracking import setup, parent_run, child_run, count_trials
from models.tracking import SEEDS, splits_sha256


@pytest.fixture
def tracking(tmp_path):
    setup(db_path=tmp_path / "t.db", artifact_dir=tmp_path / "art", experiment="t")
    return "t"


def run_one(model, tier, variant, family):
    with parent_run(model, tier, variant, family, splits_hash="abc123") as p:
        for i, combo in enumerate([(0, 1), (0, 2)]):
            with child_run(i, combo, n_train=870, n_test=299):
                mlflow.log_metrics({"dummy": float(i)})
    return p.info.run_id


@pytest.mark.parametrize("family,expected", [
    ("core", "true"), ("robustness", "true"),
    ("tuning", "false"), ("cost_grid", "false"), ("smoke", "false")])
def test_parent_tags(tracking, family, expected):
    pid = run_one("xgb", "F1", "base", family)
    tags = MlflowClient().get_run(pid).data.tags
    assert tags["trial_id"] == "xgb_F1_base"
    assert tags["trial_family"] == family
    assert tags["counts_toward_N"] == expected
    assert len(tags["git_commit"]) == 40
    assert tags["splits_sha256"] == "abc123"
    assert tags["seeds"] == "42,43,44"
    assert "data_sha_ohlcv_daily" in tags


def test_children_and_artifact(tracking):
    pid = run_one("ridge", "F0", "base", "core")
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(tracking)
    kids = client.search_runs(
        [exp.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{pid}'")
    assert len(kids) == 2
    assert sorted(k.data.tags["test_groups"] for k in kids) == ["0-1", "0-2"]
    assert "phase2_snapshot.json" in [a.path for a in client.list_artifacts(pid)]


def test_invalid_family_raises(tracking):
    with pytest.raises(ValueError):
        with parent_run("xgb", "F1", "base", "typo", splits_hash="abc"):
            pass


def test_count_trials_uses_distinct_trial_ids(tracking):
    run_one("xgb", "F1", "base", "core")
    run_one("xgb", "F1", "base", "core")
    run_one("ridge", "F0", "base", "robustness")
    run_one("xgb", "F1", "tune", "tuning")
    assert count_trials("t") == 2
    
def test_seeds_are_fixed():
    assert SEEDS == (42, 43, 44)
    
def test_splits_hash_is_pinned():
    model_pos = np.arange(19, 1212)
    h = splits_sha256(model_pos, make_groups(model_pos), make_combos(), 1217)
    assert h.startswith("b53636f3906a")