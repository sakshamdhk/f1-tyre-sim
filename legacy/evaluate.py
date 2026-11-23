import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

TABLE = "data/model_table.csv"
FOLDS = "data/folds.csv"


def load():
    t = pd.read_csv(TABLE)
    try:
        f = pd.read_csv(FOLDS)
    except FileNotFoundError:
        fold = np.zeros(len(t), dtype=int)
        for i, (_, test_idx) in enumerate(GroupKFold(n_splits=5).split(t, groups=t.Race)):
            fold[test_idx] = i
        f = pd.DataFrame({"Race": t.Race, "fold": fold}).drop_duplicates()
        f.to_csv(FOLDS, index=False)
    t["fold"] = t.Race.map(dict(zip(f.Race, f.fold)))
    return t


def evaluate(make_model, features, name="model", quiet=False, save=False):
    t = load()
    oof = np.zeros(len(t))
    os.makedirs("models", exist_ok=True)
    for k in range(5):
        tr, te = t[t.fold != k], t[t.fold == k]
        m = make_model()
        if getattr(m, "needs_groups", False):
            m.fit(tr[features], tr.lap_delta, groups=tr.Race.values)
        else:
            m.fit(tr[features], tr.lap_delta)
        oof[te.index] = m.predict(te[features])
        if save:
            joblib.dump(m, f"models/{name}_fold{k}.joblib", compress=3)
    t["pred"] = oof
    err = t.pred - t.lap_delta
    mae = err.abs().mean()
    if not quiet:
        guess = (t.lap_delta - t.lap_delta.mean()).abs().mean()
        print(f"{name}: MAE {mae:.3f}  RMSE {np.sqrt((err ** 2).mean()):.3f}  "
              f"R2 {r2_score(t.lap_delta, t.pred):.3f}  (constant-guess MAE {guess:.3f})")
        print(err.abs().groupby(t.Race).mean().round(3).sort_values().to_string())
        t[["Race", "Driver", "LapNumber", "pred"]].to_csv(f"data/oof_{name}.csv", index=False)
    return mae
