import os
import pandas as pd
from stint_eval import FEATURES
from optimize import truth, folds, offsets, load_model, slopes, optimal_pit, simulate

tr, fd, off = truth(), folds(), offsets()
names = [n for n in FEATURES if os.path.exists(f"models/stint_{n}_fold0.joblib")]
rows = []
for race, r in tr.iterrows():
    if race not in fd or not r.comparable or r.pair is None:
        continue
    n = int(r.total_laps)
    a, b = [x.lower() for x in r.pair]
    rec = {"Race": race, "pair": f"{a}-{b}", "laps": n, "actual": r.median_first_stop, "mid": n // 2}
    for name in names:
        rec[name] = optimal_pit(slopes(load_model(name, race), name, race), off, a, b, n)
    rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv("data/pit_results.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string())
for c in ["mid"] + names:
    e = res[c] - res.actual
    print(f"{c:<5} mean abs pit-lap error {e.abs().mean():.2f} laps   bias {e.mean():+.2f}")

race, a, b = res.Race.iloc[0], *res.pair.iloc[0].split("-")
n = int(res.laps.iloc[0])
sl = slopes(load_model(names[0], race), names[0], race)
sim = simulate(race, sl, a, b, optimal_pit(sl, off, a, b, n), n)
assert abs(sum(d["lap_time"] for d in sim) - sim[-1]["total"]) < 1e-6
print(f"simulate() OK: {race}, {n} laps, total {sim[-1]['total']:.1f} s")
