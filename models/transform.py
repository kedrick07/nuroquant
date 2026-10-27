# models/transform.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

RECIPE_PATH = Path("results/macro_recipe.json")
MACRO_PATH = Path("data/features/f2_macro_raw.parquet")
PANEL_PATH = Path("data/features/panel_phase2.parquet")


def load_recipe(path=RECIPE_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_macro(recipe, path=MACRO_PATH):
    """Date-indexed frame with exactly the macro columns listed in the recipe."""
    macro = pd.read_parquet(path)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()
    return macro[recipe["macro_feats"]]

class FoldMacroTransformer:
    def __init__(self, macro, recipe, clip=5.0):
        self.macro = macro                      # date-indexed raw macro (1217 x 10)
        self.recipe = recipe
        self.cols = list(recipe["macro_feats"])
        self.clip = clip
        self.mean_ = None
        self.std_ = None

    def fit(self, train_dates):
        """Fit mean/std on the training dates only (ddof=0, like StandardScaler)."""
        train = self.macro.loc[train_dates, self.cols]
        self.mean_ = train.mean()
        self.std_ = train.std(ddof=0).replace(0.0, 1.0)
        return self

    def unclipped_z(self):
        return (self.macro[self.cols] - self.mean_) / self.std_

    def transform_macro(self):
        """Scaled macro for ALL dates, clipped at +/- clip."""
        return self.unclipped_z().clip(-self.clip, self.clip)
    
    def transform(self, panel):
        """panel needs: date, ticker, and the stock-side columns (beta60, vol20).
        Returns date, ticker, 10 z_<macro> columns and 8 ix_<macro>__<stock> columns."""
        z = self.transform_macro()
        z.columns = [f"z_{c}" for c in z.columns]
        out = panel[["date", "ticker"]].merge(
            z, left_on="date", right_index=True, how="left")
        for m, s in self.recipe["interactions"]:
            out[f"ix_{m}__{s}"] = out[f"z_{m}"].to_numpy() * panel[s].to_numpy()
        return out

if __name__ == "__main__":
    from models.cpcv import load_calendar, make_groups, split_positions

    recipe = load_recipe()
    macro = load_macro(recipe)
    calendar, model_pos = load_calendar()
    groups = make_groups(model_pos)

    for combo in [(3, 4), (1, 2)]:
        train_pos, _ = split_positions(model_pos, groups, combo, len(calendar))
        tf = FoldMacroTransformer(macro, recipe).fit(calendar[train_pos])
        raw_z = tf.unclipped_z()
        z = tf.transform_macro()

        print(f"\n=== split {combo}: train rows {len(train_pos)} ===")
        print(pd.DataFrame({"mean": tf.mean_, "std": tf.std_}).round(4))
        print("clipped values per column:")
        print((raw_z.abs() > 5).sum().loc[lambda s: s > 0].to_string() or "none")
        print("max |z| after clip:", round(float(z.abs().max().max()), 3))
        for col in ["wti_chg_20", "d_unrate_21"]:
            d = raw_z[col].abs().idxmax()
            print(f"largest |z| {col}: {d.date()}  z={raw_z.loc[d, col]:.2f}")
            
    panel = pd.read_parquet(PANEL_PATH, columns=["date", "ticker", "beta60", "vol20"])
    panel["date"] = pd.to_datetime(panel["date"])
    train_pos, _ = split_positions(model_pos, groups, (3, 4), len(calendar))
    tf = FoldMacroTransformer(macro, recipe).fit(calendar[train_pos])
    out = tf.transform(panel)

    ix_cols = [c for c in out.columns if c.startswith("ix_")]
    print("\n=== panel transform, split (3, 4) ===")
    print("shape          :", out.shape)
    print("NaN cells      :", int(out.isna().sum().sum()))
    print("row order kept :", out["date"].equals(panel["date"]))
    print("n interactions :", len(ix_cols))
    print("max |interaction|:", round(float(out[ix_cols].abs().max().max()), 3))
    r = 12345
    m, s = recipe["interactions"][0]
    print("hand check     :", np.isclose(out.loc[r, f"ix_{m}__{s}"],
          out.loc[r, f"z_{m}"] * panel.loc[r, s]))