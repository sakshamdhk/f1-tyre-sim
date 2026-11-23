"""Pit-stop strategy optimiser and lap-by-lap race simulator.

Race time = sum over laps of (base pace + compound offset + tyre deg - fuel gain + track
evolution) + pit losses. Base pace, fuel gain and track evolution are the same for every strategy, so the optimiser only compares
compound offsets, predicted tyre degradation and pit losses.

The tyre model is always the one in which the race was held out (`load_model`), so no race is
ever planned by a model that was trained on it.
"""
import itertools
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from f1sim.common import MIN_STINT, fuel_k
from f1sim.features import COND, add_wear, curve_rows

COMPS = ["soft", "medium", "hard"]
WINDOW_S = 1.0     # pit laps whose race time is within this of the best one form the window


@lru_cache(maxsize=None)
def races():
    return pd.read_csv("data/races.csv", index_col="RaceId")


@lru_cache(maxsize=None)
def offsets():
    return json.load(open("data/compound_offsets.json"))


@lru_cache(maxsize=None)
def limits():
    return json.load(open("data/compound_limits.json"))


@lru_cache(maxsize=None)
def weather():
    return pd.read_csv("data/race_weather.csv", index_col=["RaceId", "LapNumber"])


@lru_cache(maxsize=None)
def load_model(name, race_id, protocol="cv"):
    r = races().loc[race_id]
    if protocol == "temporal":
        if r.Year != 2024:
            raise ValueError("the temporal protocol only predicts 2024 races")
        return joblib.load(f"models/temporal/{name}_2024.joblib")
    return joblib.load(f"models/cv/{name}_{int(r.fold)}.joblib")


def lap_conditions(race_id):
    """One row per lap of the race: track descriptors plus that lap's weather."""
    r = races().loc[race_id]
    n = int(r.total_laps)
    w = weather().loc[race_id].reindex(range(1, n + 1)).ffill().bfill()
    cond = pd.DataFrame({c: r[c] for c in ["corners", "top_speed", "ref_lap", "total_laps", "street"]},
                        index=w.index)
    cond["corner_density"] = r.corners / r.ref_lap * 60
    return pd.concat([cond, w], axis=1)


class StintCosts:
    """Predicted time lost to tyre wear for every possible stint of a race, from one batch of
    predictions: cost[c][k][s][L] = sum of deg over a stint on compound c, stint number k,
    starting on lap s, lasting L laps, on fresh tyres."""

    def __init__(self, model, race_id, member=None, override=None):
        self.race_id = race_id
        cond = lap_conditions(race_id)
        for col, value in (override or {}).items():      # what-if: e.g. {"track_temp": 45}
            cond[col] = value
        n = self.n = len(cond)
        blocks, keys = [], []
        for c, k, s in itertools.product(COMPS, (1, 2, 3), range(1, n + 1)):
            ages = np.arange(1, n - s + 2)
            rows = curve_rows({**{x: 0 for x in COND}, **cond.loc[s].to_dict()}, ages, stint_number=k,
                              start_lap=s)
            # weather follows the race: each stint lap gets the weather of its own race lap
            for col in ["track_temp", "air_temp", "humidity", "wind_speed"]:
                rows[col] = cond[col].to_numpy()[s - 1:]
            for cc in COMPS:
                rows[cc] = int(cc == c)
            blocks.append(add_wear(rows))
            keys.append((c, k, s, len(ages)))
        X = pd.concat(blocks, ignore_index=True)
        if member is not None:
            pred = model.members[member].predict(X)
        else:
            pred = model.predict(X)
        self.deg, self.cost, i = {}, {}, 0
        for c, k, s, L in keys:
            p = pred[i:i + L]
            p = p - p[0]                       # loss relative to the first lap on fresh tyres
            self.deg[c, k, s] = p
            self.cost[c, k, s] = np.concatenate([[0.0], np.cumsum(p)])
            i += L

    def stint(self, c, k, start, length):
        if length > limits()[c]:
            return np.inf             # longer than any stint seen in the data on this compound
        return self.cost[c, min(k, 3), start][length] + offsets()[c] * length


def legal(comps):
    return len(set(comps)) >= 2          # dry races must use at least two compounds


def evaluate_strategy(sc, comps, stops, pit_loss):
    bounds = [1, *[p + 1 for p in stops], sc.n + 1]
    return sum(sc.stint(c, i + 1, bounds[i], bounds[i + 1] - bounds[i]) for i, c in enumerate(comps)) \
        + pit_loss * len(stops)


def one_stop(sc, a, b, pit_loss):
    """Race time (relative) for every first-stint length p, compounds a then b."""
    ps = np.arange(MIN_STINT, sc.n - MIN_STINT + 1)
    return ps, np.array([evaluate_strategy(sc, (a, b), [p], pit_loss) for p in ps])


def best_strategies(sc, pit_loss, max_stops=2):
    """Best 1-stop and 2-stop strategy for every legal compound order, sorted by race time."""
    out = []
    for a, b in itertools.permutations(COMPS, 2):
        ps, cost = one_stop(sc, a, b, pit_loss)
        i = int(np.argmin(cost))
        if np.isfinite(cost[i]):
            out.append(dict(comps=(a, b), stops=(int(ps[i]),), time=float(cost[i])))
    if max_stops >= 2:
        n = sc.n
        for comps in itertools.product(COMPS, repeat=3):
            if not legal(comps):
                continue
            best = None
            for p1 in range(MIN_STINT, n - 2 * MIN_STINT + 1):
                for p2 in range(p1 + MIN_STINT, n - MIN_STINT + 1):
                    t = evaluate_strategy(sc, comps, [p1, p2], pit_loss)
                    if best is None or t < best[0]:
                        best = (t, (p1, p2))
            if np.isfinite(best[0]):
                out.append(dict(comps=comps, stops=best[1], time=float(best[0])))
    return sorted(out, key=lambda s: s["time"])


def pit_window(sc, a, b, pit_loss):
    ps, cost = one_stop(sc, a, b, pit_loss)
    best = int(ps[np.argmin(cost)])
    ok = ps[cost <= cost.min() + WINDOW_S]
    return best, int(ok.min()), int(ok.max())


def parse_strategy(text):
    """'M-H:22' -> (('medium', 'hard'), (22,)); 'S-M-H:15,40' for two stops."""
    names = {"s": "soft", "m": "medium", "h": "hard"}
    comps, laps = text.lower().split(":")
    comps = tuple(names[c[0]] for c in comps.split("-"))
    stops = tuple(int(x) for x in laps.split(","))
    if len(stops) != len(comps) - 1:
        raise ValueError("need one pit lap per compound change, e.g. M-H:22 or S-M-H:15,40")
    return comps, stops


def simulate(race_id, comps, stops, sc, window=None):
    """Lap-by-lap race on one strategy. Returns a list of dicts; the sum of lap_time is total."""
    r = races().loc[race_id]
    k, off = fuel_k(r.total_laps), offsets()
    bounds = [1, *[p + 1 for p in stops], sc.n + 1]
    out, total = [], 0.0
    for i, c in enumerate(comps):
        start = bounds[i]
        for lap in range(start, bounds[i + 1]):
            age = lap - start + 1
            base = r.base_pace + off[c]
            deg = float(sc.deg[c, min(i + 1, 3), start][age - 1])
            fuel = -k * lap
            track = r.evo * lap               # track rubbering in (negative = getting faster)
            pit = float(r.pit_loss) if lap in stops else 0.0
            lap_time = base + deg + fuel + track + pit
            total += lap_time
            out.append(dict(lap=lap, compound=c, tyre_age=age, base=base, deg=deg, fuel=fuel, track=track, pit=pit,
                            lap_time=lap_time, total=total,
                            window=window is not None and window[0] <= lap <= window[1]))
    return out
