"""Tuned XGBoost over CPCV splits, logged to MLflow."""
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import xgboost

from models.cpcv import (
    load_calendar, make_combos, make_groups, split_positions,
)
from models.runner import (
    RET, TARGET, load_panel, rank_ic_by_date, split_frames,
    tier_features, weekly_eval_dates,
)
from models.tracking import child_run, parent_run, setup
from models.tuning import (
    XGB_FIXED, XGB_GRID, XGB_SEEDS, select_xgb_config, xgb_model,
)


def config_name(c):
    return f"d{c['max_depth']}_n{c['n_estimators']}_w{c['min_child_weight']}"


def run_xgb(tier, family="core", variant="main", splits=None):
    if tier == "F2":
        raise NotImplementedError(
            "F2 needs the fold-wise macro transformer inside each inner fold"
        )
    panel = load_panel()
    mr = panel[panel["model_row"]]
    calendar, model_pos = load_calendar()
    groups = make_groups(model_pos)
    combos = make_combos()
    weekly = weekly_eval_dates(calendar, model_pos)
    features = tier_features(tier)
    splits = list(range(len(combos))) if splits is None else list(splits)

    setup()
    split_ics = []
    with parent_run("xgboost", tier, variant, family,
                    seeds=XGB_SEEDS) as parent:
        mlflow.log_params({
            "xgboost_version": xgboost.__version__,
            "fixed_params": str(XGB_FIXED),
            "grid": str([config_name(c) for c in XGB_GRID]),
            "inner_folds": 4,
            "selection_metric": "mean_weekly_rank_ic",
            "target": TARGET,
            "n_features": len(features),
            "n_splits": len(splits),
        })
        mlflow.log_dict({"features": features}, "features.json")

        for s in splits:
            combo = combos[s]
            train_pos, test_pos = split_positions(
                model_pos, groups, combo, len(calendar)
            )
            train, test = split_frames(panel, calendar, train_pos, test_pos)

            with child_run(s, combo, n_train=len(train), n_test=len(test)):
                config, inner = select_xgb_config(
                    mr, calendar, train_pos, features, weekly
                )
                seed_preds = []
                for seed in XGB_SEEDS:
                    model = xgb_model(config, seed)
                    model.fit(train[features], train[TARGET])
                    seed_preds.append(model.predict(test[features]))

                preds = test[["date", "ticker", TARGET, RET]].copy()
                preds["pred"] = np.mean(seed_preds, axis=0)
                preds["split_idx"] = s
                assert np.isfinite(preds["pred"]).all()
                assert not preds.duplicated(["date", "ticker"]).any()

                wk = preds[preds["date"].isin(weekly)]
                assert wk.groupby("date").size().eq(30).all()
                ic = rank_ic_by_date(wk)
                assert np.isfinite(ic.to_numpy()).all()
                split_ics.append(float(ic.mean()))

                mlflow.log_params({
                    "selected_config": config_name(config),
                    "selected_max_depth": config["max_depth"],
                    "selected_n_estimators": config["n_estimators"],
                    "selected_min_child_weight": config["min_child_weight"],
                })
                mlflow.log_metrics({
                    "mean_weekly_ic": float(ic.mean()),
                    "n_weekly_dates": int(len(ic)),
                    "inner_best_ic": float(inner.max()),
                })
                mlflow.log_dict(
                    {config_name(XGB_GRID[i]): float(v)
                     for i, v in inner.items()},
                    "inner_ic.json",
                )
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "predictions.parquet"
                    preds.to_parquet(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="predictions")

            print(
                f"{tier} split {s:02d} {combo}: {config_name(config)}, "
                f"weekly dates={len(ic)}, weekly IC={ic.mean():+.4f}"
            )

        mlflow.log_metric("mean_split_weekly_ic", float(np.mean(split_ics)))
    return parent.info.run_id


if __name__ == "__main__":
    tier, family, variant = sys.argv[1:4]
    subset = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    print("parent run:", run_xgb(tier, family, variant, subset))
