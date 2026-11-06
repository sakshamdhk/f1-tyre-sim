import json
import pandas as pd
from scipy.stats import theilslopes
from sklearn.linear_model import LinearRegression
from evaluate import load

t = load()
c = pd.read_csv("data/clean_laps.csv", usecols=["Race", "Driver", "LapNumber", "corrected"])
t = t.merge(c, on=["Race", "Driver", "LapNumber"], how="left")
t["comp"] = t[["soft", "medium", "hard"]].idxmax(axis=1)

rows = []
for (race, drv, st), g in t.groupby(["Race", "Driver", "stint_number"]):
    if len(g) < 10 or g.tyre_age.max() - g.tyre_age.min() < 8:
        continue
    slope, icpt = theilslopes(g.corrected, g.tyre_age)[:2]
    rows.append(dict(Race=race, Driver=drv, stint=st, comp=g.comp.iloc[0], n=len(g),
                     slope=slope, pace2=icpt + 2 * slope, start_lap=g.LapNumber.min(),
                     track_temp=g.track_temp.mean(), air_temp=g.air_temp.mean(),
                     humidity=g.humidity.mean(), fold=g.fold.iloc[0]))
s = pd.DataFrame(rows)
s = s[s.slope.between(-0.1, 0.4)].reset_index(drop=True)
for comp in ["soft", "medium", "hard"]:
    s[comp] = (s.comp == comp).astype(int)

# compound pace offsets: stint pace at tyre age 2, with a fixed effect per driver-in-race
# and a start-lap term for track evolution. hard = 0, so a negative offset means faster.
fx = pd.get_dummies(s.Race + "|" + s.Driver, dtype=float)
X = pd.concat([fx, s[["soft", "medium", "start_lap"]].astype(float)], axis=1)
X.columns = X.columns.astype(str)
coef = pd.Series(LinearRegression(fit_intercept=False).fit(X, s.pace2).coef_, index=X.columns)
off = {"soft": float(coef["soft"]), "medium": float(coef["medium"]), "hard": 0.0}
print("estimated compound offsets (s, vs hard):", {k: round(v, 3) for k, v in off.items()})
if not (-2.5 < off["soft"] < off["medium"] < 0.0):
    off = {"soft": -0.6, "medium": -0.3, "hard": 0.0}
    print("not plausible, using fallback assumption:", off)

json.dump(off, open("data/compound_offsets.json", "w"))
s.to_csv("data/stints.csv", index=False)
print(f"{len(s)} stints in {s.Race.nunique()} races")
print(s.groupby("comp").slope.agg(["count", "mean", "std"]).round(3).to_string())
