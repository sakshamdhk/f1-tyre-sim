import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from common import FEATURES
from evaluate import evaluate, load

t = load()
y = t.lap_delta
print(f"constant-guess MAE          {(y - y.mean()).abs().mean():.3f}")
print(f"variance that is race-level {r2_score(y, y.groupby(t.Race).transform('mean')):.3f}")


def demean(s):
    return s - s.groupby(t.Race).transform("mean")


print(f"within-race corr(tyre_age, lap_delta) {np.corrcoef(demean(t.tyre_age), demean(y))[0, 1]:.3f}")

comp = ["tyre_age", "soft", "medium", "hard", "age_soft", "age_medium", "age_hard"]
sets = {
    "tyre_age only": ["tyre_age"],
    "tyre + compound": comp,
    "+ stint": comp + ["stint_number"],
    "+ track_temp": comp + ["stint_number", "track_temp"],
    "+ all weather": FEATURES["lr"],
}
for name, f in sets.items():
    print(f"{name:<18} OOF MAE {evaluate(LinearRegression, f, 'tmp', quiet=True):.3f}")

f = FEATURES["lr"]
print(f"in-sample R2, all features  {LinearRegression().fit(t[f], y).score(t[f], y):.3f}")
