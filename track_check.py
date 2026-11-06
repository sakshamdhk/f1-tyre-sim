import numpy as np
import pandas as pd

s = pd.read_csv("data/stints.csv")
other_races = np.zeros(len(s))
other_drivers = np.full(len(s), np.nan)
for i, r in s.iterrows():
    other_races[i] = s[(s.comp == r.comp) & (s.Race != r.Race)].slope.mean()
    same = s[(s.comp == r.comp) & (s.Race == r.Race) & (s.Driver != r.Driver)]
    if len(same) >= 2:
        other_drivers[i] = same.slope.mean()
ok = ~np.isnan(other_drivers)
print(f"{ok.sum()} stints with at least 2 other drivers on the same compound at the same race")
print(f"predict from the same compound at OTHER races:            MAE {np.abs(other_races - s.slope)[ok].mean():.4f}")
print(f"predict from the same compound, same race, other drivers: MAE {np.abs(other_drivers - s.slope)[ok].mean():.4f}")
