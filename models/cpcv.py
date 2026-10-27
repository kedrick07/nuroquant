from pathlib import Path

from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd

PANEL_PATH = Path("data/features/panel_phase2.parquet")

def load_calendar(path=PANEL_PATH):
    """
    Return (calendar, model_pos).
    calendar: sorted DatetimeIndex of all trading dates in the panel.
    model_pos: integer positions in 'calendar' where the model_row is True/
    """
    panel = pd.read_parquet(path, columns=["date", "model_row"])
    panel["date"] = pd.to_datetime(panel["date"])

    # model_row repeats for all 30 tickers on a date, so it must agree within a date
    assert panel.groupby("date")["model_row"].nunique().max() == 1

    per_date = panel.groupby("date")["model_row"].any().sort_index()
    calendar = pd.DatetimeIndex(per_date.index)
    model_pos = np.flatnonzero(per_date.to_numpy())
    return calendar, model_pos

def make_groups(model_pos, n_groups=8):
    """Split model positions into n contiguous blocks (sizes differ by at most 1)."""
    return np.array_split(model_pos, n_groups)


def make_combos(n_groups=8, k=2):
    """All ways to pick k test groups out of n_groups."""
    return list(combinations(range(n_groups), k))

def split_positions(model_pos, groups, test_groups, n_calendar,
                    horizon=5, embargo=20):
    """Return (train_pos, test_pos) as calendar positions for one split."""
    test_pos = np.concatenate([groups[g] for g in test_groups])

    blocked = np.zeros(n_calendar, dtype=bool)
    for t in test_pos:
        blocked[max(0, t - horizon): t + embargo + 1] = True

    train_pos = model_pos[~blocked[model_pos]]
    return train_pos, test_pos

def assign_paths(combos, n_groups=8):
    """paths[j][g] = index of the split whose model predicts group g on path j."""
    k = len(combos[0])
    n_paths = len(combos) * k // n_groups          # 28 * 2 / 8 = 7
    seen = [0] * n_groups
    paths = [dict() for _ in range(n_paths)]
    for s, combo in enumerate(combos):
        for g in combo:
            paths[seen[g]][g] = s
            seen[g] += 1
    return paths

if __name__ == "__main__":
    calendar, model_pos = load_calendar()
    groups = make_groups(model_pos)
    for i, g in enumerate(groups):
        print(f"group {i}: n={len(g):3d}  pos {g[0]:4d}->{g[-1]:4d}  "
              f"{calendar[g[0]].date()} -> {calendar[g[-1]].date()}")

    combos = make_combos()
    print("n splits:", len(combos))
    print("first 3 :", combos[:3])
    print("appearances per group:", dict(sorted(Counter(i for c in combos for i in c).items())))
    
    for combo in [(0, 1), (3, 4), (0, 7)]:
        tr, te = split_positions(model_pos, groups, combo, len(calendar))
        print(combo, "train", len(tr), "test", len(te),
              "| train pos", tr.min(), "->", tr.max())
    
    paths = assign_paths(combos)
    print("n paths:", len(paths))
    print("path 0 :", paths[0])
    print("groups per path:", {len(p) for p in paths})
    used = Counter((s, g) for p in paths for g, s in p.items())
    print("pairs used:", len(used), "| max reuse:", max(used.values()))