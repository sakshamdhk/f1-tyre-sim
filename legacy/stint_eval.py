import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

C = ["soft", "medium", "hard"]
W = ["track_temp", "air_temp", "humidity"]
FEATURES = {"mean": C, "lr": C + W, "rf": C + W, "nn": C + W}


class CompMean:
    """Baseline: every stint gets the average slope of its compound."""

    def fit(self, X, y):
        self.m = {c: float(y[X[c] == 1].mean()) if (X[c] == 1).any() else float(y.mean()) for c in C}
        return self

    def predict(self, X):
        return sum(X[c].to_numpy() * self.m[c] for c in C)


def evaluate_stints(make, name, save=False):
    s = pd.read_csv("data/stints.csv")
    f = FEATURES[name]
    p = np.zeros(len(s))
    os.makedirs("models", exist_ok=True)
    for k in range(5):
        tr, te = s[s.fold != k], s[s.fold == k]
        m = make()
        if getattr(m, "needs_groups", False):
            m.fit(tr[f], tr.slope, groups=tr.Race.values)
        else:
            m.fit(tr[f], tr.slope)
        if len(te):
            p[te.index] = m.predict(te[f])
        if save:
            joblib.dump(m, f"models/stint_{name}_fold{k}.joblib", compress=3)
    mae = np.abs(p - s.slope).mean()
    print(f"{name:<5} MAE {mae:.4f} s/lap   R2 {r2_score(s.slope, p):.3f}")
    s.assign(pred=p)[["Race", "Driver", "stint", "pred"]].to_csv(f"data/stint_oof_{name}.csv", index=False)
    return mae
