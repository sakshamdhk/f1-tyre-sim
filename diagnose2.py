import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from evaluate import load

t = load()
c = pd.read_csv("data/clean_laps.csv", usecols=["Race", "Driver", "LapNumber", "corrected"])
t = t.merge(c, on=["Race", "Driver", "LapNumber"], how="left")
t = t.sort_values(["Race", "Driver", "LapNumber"]).reset_index(drop=True)

# new target: lap time relative to the start of the SAME stint (median of its first 3 clean laps)
g = t.groupby(["Race", "Driver", "stint_number"])
start = g.corrected.transform(lambda s: s.iloc[:3].median())
size = g.corrected.transform("size")
t["target"] = t.corrected - start
t = t[size >= 8].reset_index(drop=True)

print(t.groupby(pd.cut(t.tyre_age, [0, 5, 10, 15, 20, 30, 80]), observed=True).target.mean().round(3).to_string())

FEAT = ["tyre_age", "soft", "medium", "hard", "age_soft", "age_medium", "age_hard", "stint_number"]


def oof_mae(make, feats):
    pred = np.zeros(len(t))
    for k in range(5):
        tr, te = t[t.fold != k], t[t.fold == k]
        pred[te.index] = make().fit(tr[feats], tr.target).predict(te[feats])
    return np.abs(pred - t.target).mean()


rf = lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=42)
print(f"constant-guess MAE  {(t.target - t.target.mean()).abs().mean():.3f}")
print(f"LR  tyre only       {oof_mae(LinearRegression, FEAT):.3f}")
print(f"LR  + track_temp    {oof_mae(LinearRegression, FEAT + ['track_temp']):.3f}")
print(f"RF  tyre only       {oof_mae(rf, FEAT):.3f}")
print(f"RF  + track_temp    {oof_mae(rf, FEAT + ['track_temp']):.3f}")
