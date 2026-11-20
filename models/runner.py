# models/runner.py
import tempfile
import mlflow
from sklearn.linear_model import Ridge
from models.tracking import setup, parent_run, child_run
from pathlib import Path
import numpy as np
import pandas as pd
from models.cpcv import load_calendar, make_groups, make_combos, split_positions

PANEL_PATH = Path("data/features/panel_phase2.parquet")

F0 = ["ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "sma20_ratio",
      "sma50_ratio", "macd_norm", "rsi14", "vol20", "logvol_ratio", "beta60"]
F1_EXTRA = ["sent_mean", "sent_ewm5", "sent_mom", "abn_count",
            "days_since_news", "has_news"]
TARGET, RET = "target_rank", "target_ret5"


def tier_features(tier, recipe=None):
    if tier == "F0":
        return list(F0)
    f1 = F0 + F1_EXTRA
    if tier == "F1":
        return f1
    if tier == "F2":
        z = [f"z_{c}" for c in recipe["macro_feats"]]
        ix = [f"ix_{m}__{s}" for m, s in recipe["interactions"]]
        return f1 + z + ix
    raise ValueError(f"unknown tier {tier!r}")


def load_panel(path=PANEL_PATH):
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)

def split_frames(panel, calendar, train_pos, test_pos):
    """Model rows for the training and test dates of one CPCV split"""
    mr = panel[panel["model_row"]]
    tr = mr[mr["date"].isin(calendar[train_pos])]
    te = mr[mr["date"].isin(calendar[test_pos])]
    return tr, te

def rank_ic_by_date(df, pred="pred", y=RET):
    """Spearmen correlation between rpedictions and realised return, per date"""
    out ={}
    for d, g in df.groupby("date"):
        out[d] = g[pred].rank().corr(g[y].rank())
    return pd.Series(out, name="rank_ic")

def ridge_smoke_test():
    panel = load_panel()
    calendar, model_pos = load_calendar()
    groups = make_groups(model_pos)
    combos = make_combos()

    combo = (3, 4)
    split_idx = combos.index(combo)
    train_pos, test_pos = split_positions(
        model_pos, groups, combo, len(calendar)
    )
    train, test = split_frames(panel, calendar, train_pos, test_pos)

    setup()

    for tier in ("F0", "F1"):
        features = tier_features(tier)

        with parent_run(
            model="ridge",
            tier=tier,
            variant="smoke_alpha1",
            family="smoke",
            seeds=(),
        ) as parent:
            mlflow.log_params({
                "alpha": 1.0,
                "solver": "svd",
                "target": TARGET,
                "n_features": len(features),
                "evaluation": "daily_diagnostic_only",
            })
            mlflow.log_dict({"features": features}, "features.json")

            with child_run(
                split_idx,
                combo,
                n_train=len(train),
                n_test=len(test),
            ):
                model = Ridge(alpha=1.0, solver="svd")
                model.fit(train[features], train[TARGET])

                predictions = test[
                    ["date", "ticker", TARGET, RET]
                ].copy()
                predictions["pred"] = model.predict(test[features])
                predictions["split_idx"] = split_idx

                assert len(predictions) == len(test)
                assert np.isfinite(predictions["pred"]).all()
                assert not predictions.duplicated(
                    ["date", "ticker"]
                ).any()

                daily_ic = rank_ic_by_date(predictions)
                assert np.isfinite(daily_ic.to_numpy()).all()

                mlflow.log_metrics({
                    "diagnostic_daily_mean_ic": float(daily_ic.mean()),
                    "prediction_std": float(predictions["pred"].std()),
                })

                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "predictions.parquet"
                    predictions.to_parquet(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="predictions")

                print(
                    f"{tier}: train={len(train)}, test={len(test)}, "
                    f"features={len(features)}, "
                    f"finite predictions=True, "
                    f"daily diagnostic IC={daily_ic.mean():.4f}"
                )

            print("parent run:", parent.info.run_id)

def weekly_eval_dates(calendar, model_pos):
    """Global 5-trading-day schedule, anchored at the first model date."""
    calendar = pd.DatetimeIndex(calendar)
    positions = np.asarray(model_pos)

    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("model_pos must be a non-empty 1D array")
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("model_pos must contain integer positions")
    if positions[0] < 0 or positions[-1] >= len(calendar):
        raise ValueError("model_pos is outside the calendar")
    if not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("calendar must be sorted and unique")
    if not np.all(np.diff(positions) == 1):
        raise ValueError("model dates must be consecutive trading days")

    return calendar[positions[::5]]

if __name__ == "__main__":
    from models.transform import load_recipe

    recipe = load_recipe()
    panel = load_panel()
    mr = panel[panel["model_row"]]
    f1 = tier_features("F1")

    print("panel shape      :", panel.shape)
    print("model rows       :", len(mr))
    print("n features F0/F1/F2:", len(tier_features("F0")), len(f1),
          len(tier_features("F2", recipe)))
    print("NaN in F1 (model rows):", int(mr[f1].isna().sum().sum()))
    print("NaN in targets        :", int(mr[[TARGET, RET]].isna().sum().sum()))
    print("max |value| per feature:")
    print(mr[f1].abs().max().round(3).to_string())
    
    calendar, model_pos = load_calendar()
    groups = make_groups(model_pos)
    train_pos, test_pos = split_positions(model_pos, groups, (3, 4), len(calendar))
    tr, te = split_frames(panel, calendar, train_pos, test_pos)
    print("\nsplit (3, 4): train rows", len(tr), "| test rows", len(te))
    print("train max date < test min date? ", tr["date"].max() < te["date"].min(),
          "(False expected: group 4 sits mid-sample)")
    print("date overlap:", len(set(tr["date"]) & set(te["date"])))

    te = te.copy()
    rng = np.random.default_rng(0)
    te["pred"] = te[RET]
    print("IC, perfect pred :", round(rank_ic_by_date(te).mean(), 4))
    te["pred"] = -te[RET]
    print("IC, reversed pred:", round(rank_ic_by_date(te).mean(), 4))
    te["pred"] = rng.normal(size=len(te))
    print("IC, random pred  :", round(rank_ic_by_date(te).mean(), 4))