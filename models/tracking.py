# models/tracking.py
import hashlib
import json
import subprocess
import contextlib
import mlflow
from pathlib import Path


SNAPSHOT_PATH = Path("results/phase2_snapshot.json")
SEEDS = (42, 43, 44)
EXPERIMENT = "nuroquant"
FAMILY_COUNTS = {"core": True, "robustness": True,
                 "tuning": False, "cost_grid": False, "smoke": False}


def git_info():
    """(commit hash, dirty flag). dirty = any uncommitted or untracked change."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                            text=True, check=True).stdout.strip()
    return commit, bool(status)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_checksums(path=SNAPSHOT_PATH):
    """{file stem: first 12 hex chars}, same style as the Phase 2 report."""
    with open(path, encoding="utf-8") as f:
        files = json.load(f)["files"]
    return {Path(p).stem: h[:12] for p, h in files.items()}


def splits_sha256(model_pos, groups, combos, n_calendar):
    """One hash over every split's combo, train positions and test positions."""
    from models.cpcv import split_positions
    h = hashlib.sha256()
    for combo in combos:
        tr, te = split_positions(model_pos, groups, combo, n_calendar)
        h.update(repr(combo).encode())
        h.update(tr.astype("int64").tobytes())
        h.update(te.astype("int64").tobytes())
    return h.hexdigest()

def setup(db_path="mlflow.db", artifact_dir="mlruns", experiment=EXPERIMENT):
    """Point MLflow at a SQLite file and make sure the experiment exists."""
    mlflow.set_tracking_uri(f"sqlite:///{Path(db_path).resolve()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=Path(artifact_dir).resolve().as_uri())
    mlflow.set_experiment(experiment)


def current_splits_hash():
    from models.cpcv import load_calendar, make_groups, make_combos
    calendar, model_pos = load_calendar()
    return splits_sha256(model_pos, make_groups(model_pos), make_combos(), len(calendar))


@contextlib.contextmanager
def parent_run(model, tier, variant, family, splits_hash=None,
               seeds=SEEDS, extra_tags=None):
    if family not in FAMILY_COUNTS:
        raise ValueError(f"unknown trial_family {family!r}; use {list(FAMILY_COUNTS)}")
    commit, dirty = git_info()
    trial_id = f"{model}_{tier}_{variant}"
    tags = {
        "trial_id": trial_id,
        "trial_family": family,
        "counts_toward_N": str(FAMILY_COUNTS[family]).lower(),
        "model": model, "tier": tier, "variant": variant,
        "git_commit": commit, "git_dirty": str(dirty).lower(),
        "data_snapshot_sha256": file_sha256(SNAPSHOT_PATH),
        "splits_sha256": splits_hash or current_splits_hash(),
        "seeds": ",".join(map(str, seeds)),
    }
    tags.update({f"data_sha_{k}": v for k, v in data_checksums().items()})
    tags.update(extra_tags or {})
    with mlflow.start_run(run_name=trial_id) as run:
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(SNAPSHOT_PATH))
        yield run


@contextlib.contextmanager
def child_run(split_idx, combo, n_train=None, n_test=None):
    with mlflow.start_run(run_name=f"split_{split_idx:02d}", nested=True) as run:
        mlflow.set_tags({"split_idx": str(split_idx),
                         "test_groups": "-".join(map(str, combo))})
        if n_train is not None and n_test is not None:
            mlflow.log_params({"n_train": n_train, "n_test": n_test})
        yield run


def count_trials(experiment=EXPERIMENT):
    """N for the deflated Sharpe: distinct trial_ids whose runs count toward N."""
    exp = mlflow.get_experiment_by_name(experiment)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.counts_toward_N = 'true'", output_format="list")
    return len({r.data.tags["trial_id"] for r in runs})


if __name__ == "__main__":
    from models.cpcv import load_calendar, make_groups, make_combos

    commit, dirty = git_info()
    print("git commit    :", commit[:10], "| dirty:", dirty)
    print("snapshot sha  :", file_sha256(SNAPSHOT_PATH)[:12])
    print("data checksums:", data_checksums())

    calendar, model_pos = load_calendar()
    groups, combos = make_groups(model_pos), make_combos()
    a = splits_sha256(model_pos, groups, combos, len(calendar))
    b = splits_sha256(model_pos, groups, combos, len(calendar))
    print("splits sha    :", a[:12], "| repeatable:", a == b)
    print("seeds         :", SEEDS)