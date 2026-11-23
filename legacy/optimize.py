import ast
import json
import joblib
import numpy as np
import pandas as pd
from common import K, PIT_LOSS, MIN_STINT
from stint_eval import FEATURES, C

_c = {}


def _get(key, fn):
    if key not in _c:
        _c[key] = fn()
    return _c[key]


def folds():
    f = pd.read_csv("data/folds.csv")
    return dict(zip(f.Race, f.fold))


def truth():
    p = pd.read_csv("data/pit_truth.csv")
    p["pair"] = p.compound_pair.apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else None)
    return p.set_index("Race")


def offsets():
    return _get("off", lambda: json.load(open("data/compound_offsets.json")))


def load_model(name, race):
    k = folds()[race]
    return _get((name, k), lambda: joblib.load(f"models/stint_{name}_fold{k}.joblib"))


def slopes(model, name, race):
    """Predicted degradation (s per lap of tyre age) for each compound under this race's mean conditions."""
    t = _get("table", lambda: pd.read_csv("data/model_table.csv", usecols=["Race", "track_temp", "air_temp", "humidity"]))
    cond = t[t.Race == race][["track_temp", "air_temp", "humidity"]].mean()
    X = pd.DataFrame({**{c: [int(c == k) for k in C] for c in C}, **{w: [cond[w]] * 3 for w in cond.index}})
    return {c: max(0.0, float(v)) for c, v in zip(C, model.predict(X[FEATURES[name]]))}


def optimal_pit(sl, off, a, b, n):
    p = np.arange(MIN_STINT, n - MIN_STINT + 1)
    q = n - p
    cost = off[a] * p + sl[a] * p * (p - 1) / 2 + off[b] * q + sl[b] * q * (q - 1) / 2 + PIT_LOSS
    return int(p[np.argmin(cost)])


def race_ref(race):
    s = _get("stints", lambda: pd.read_csv("data/stints.csv"))
    off = offsets()
    r = s[s.Race == race]
    r = r if len(r) else s
    return float((r.pace2 - r.comp.map(off) - r.slope).median())


def simulate(race, sl, a, b, p, n, window=None):
    off, ref = offsets(), race_ref(race)
    out, total = [], 0.0
    for lap in range(1, n + 1):
        comp = a if lap <= p else b
        age = lap if lap <= p else lap - p
        base, deg, fuel = ref + off[comp], sl[comp] * (age - 1), -K * lap
        pit = PIT_LOSS if lap == p else 0.0
        lap_time = base + deg + fuel + pit
        total += lap_time
        out.append(dict(lap=lap, compound=comp, tyre_age=age, base=base, deg=deg, fuel=fuel, pit=pit,
                        lap_time=lap_time, total=total, window=window is not None and abs(lap - window) <= 2))
    return out
