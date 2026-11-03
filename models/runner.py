# models/runner.py
from pathlib import Path

import pandas as pd

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