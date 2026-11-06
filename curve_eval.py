import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from evaluate import load

t = load()
c = pd.read_csv("data/clean_laps.csv", usecols=["Race", "Driver", "LapNumber", "corrected"])
t = t.merge(c, on=["Race", "Driver", "LapNumber"], how="left")
t = t.sort_values(["Race", "Driver", "LapNumber"]).reset_index(drop=True)
g = t.groupby(["Race", "Driver", "stint_number"])
t["target"] = t.corrected - g.corrected.transform(lambda s: s.iloc[:3].median())
t = t[g.corrected.transform("size") >= 8].reset_index(drop=True)
t["comp"] = t[["soft", "medium", "hard"]].idxmax(axis=1)

MIN_N = 3  # laps needed in a bin to trust its average
FEAT = ["tyre_age", "soft", "medium", "hard", "age_soft", "age_medium", "age_hard", "stint_number"]


def curve_scores(make, feats):
    pred = np.zeros(len(t))
    if make is not None:
        for k in range(5):
            tr, te = t[t.fold != k], t[t.fold == k]
            pred[te.index] = make().fit(tr[feats], tr.target).predict(te[feats])
    else:
        pred[:] = t.target.mean()
    b = t.assign(pred=pred).groupby(["Race", "comp", "tyre_age"]).agg(
        y=("target", "mean"), p=("pred", "mean"), n=("target", "size"))
    b = b[b.n >= MIN_N]
    return np.abs(b.p - b.y).mean(), r2_score(b.y, b.p), len(b)


rf = lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=42)
rows = {
    "constant guess": (None, FEAT),
    "LR tyre only": (LinearRegression, FEAT),
    "LR + track_temp": (LinearRegression, FEAT + ["track_temp"]),
    "RF tyre only": (rf, FEAT),
    "RF + track_temp": (rf, FEAT + ["track_temp"]),
}
print("curve-level scores (lap noise averaged out per race/compound/tyre age)")
for name, (make, feats) in rows.items():
    mae, r2, n = curve_scores(make, feats)
    print(f"{name:<18} MAE {mae:.3f}  R2 {r2:.3f}  ({n} bins)")
