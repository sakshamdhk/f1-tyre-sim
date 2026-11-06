import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from evaluate import load

t = load()
c = pd.read_csv("data/clean_laps.csv", usecols=["Race", "Driver", "LapNumber", "corrected"])
t = t.merge(c, on=["Race", "Driver", "LapNumber"], how="left")
t["comp"] = t[["soft", "medium", "hard"]].idxmax(axis=1)

rows = []
for (race, drv, st), g in t.groupby(["Race", "Driver", "stint_number"]):
    if len(g) < 10 or g.tyre_age.max() - g.tyre_age.min() < 8:
        continue
    rows.append(dict(Race=race, Driver=drv, stint=st, comp=g.comp.iloc[0], n=len(g),
                     slope=theilslopes(g.corrected, g.tyre_age)[0],
                     track_temp=g.track_temp.mean(), air_temp=g.air_temp.mean(),
                     humidity=g.humidity.mean(), fold=g.fold.iloc[0]))
s = pd.DataFrame(rows)
s = s[s.slope.between(-0.1, 0.4)].reset_index(drop=True)
s.to_csv("data/stint_slopes.csv", index=False)

print(f"{len(s)} stints")
print(s.groupby("comp").slope.agg(["count", "mean", "median", "std"]).round(3).to_string())
for comp, g in s.groupby("comp"):
    print(f"corr(slope, track_temp) {comp}: {g.slope.corr(g.track_temp):+.2f}")

for comp in ["soft", "medium", "hard"]:
    s[comp] = (s.comp == comp).astype(int)
X1 = ["soft", "medium", "hard"]
X2 = X1 + ["track_temp"]
X3 = X2 + ["air_temp", "humidity"]


def oof(make, feats):
    p = np.zeros(len(s))
    for k in range(5):
        tr, te = s[s.fold != k], s[s.fold == k]
        p[te.index] = make().fit(tr[feats], tr.slope).predict(te[feats])
    return np.abs(p - s.slope).mean(), r2_score(s.slope, p)


guess = np.zeros(len(s))
for k in range(5):
    tr, te = s[s.fold != k], s[s.fold == k]
    guess[te.index] = te.comp.map(tr.groupby("comp").slope.mean())
print("\nout-of-fold, slope in s per lap")
print(f"{'compound mean only':<20} MAE {np.abs(guess - s.slope).mean():.4f}  R2 {r2_score(s.slope, guess):.3f}")
rf = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42)
for name, make, f in [("LR compound+track", LinearRegression, X2), ("LR + air/humidity", LinearRegression, X3),
                      ("RF compound+track", rf, X2), ("RF + air/humidity", rf, X3)]:
    mae, r2 = oof(make, f)
    print(f"{name:<20} MAE {mae:.4f}  R2 {r2:.3f}")
