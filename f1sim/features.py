"""Feature sets shared by all models.

Every model predicts `deg`: the time a lap has lost to tyre wear since the start of its stint.
The stint's reference pace is the median of its first REF_LAPS laps, i.e. tyre age
`stint_start_age + 1`, so wear is measured relative to that age (u1, u2 below).
"""
import numpy as np
import pandas as pd

from f1sim.common import REF_LAPS

COMPOUND_COLS = ["soft", "medium", "hard"]
WEATHER = ["track_temp", "air_temp", "humidity", "wind_speed"]
TRACK = ["corners", "corner_density", "top_speed", "ref_lap", "total_laps", "street"]
# conditions: everything that describes the race and the stint, but not the lap within it
COND = COMPOUND_COLS + WEATHER + TRACK + ["stint_number", "stint_start_frac"]
# physically motivated subset: weather, circuit type and stint context, without the descriptors
# (corners, lap time, race length) that act as a fingerprint of each training circuit
PHYS = COMPOUND_COLS + ["track_temp", "air_temp", "humidity", "street", "stint_number", "stint_start_frac"]
FEATURE_SETS = {"phys": PHYS, "full": COND}
AGE = ["tyre_age", "stint_lap", "stint_start_age"]
LAP = COND + AGE + ["wear_lin", "wear_quad", "in_traffic"]


def wear_terms(d):
    """Linear and quadratic wear since the reference lap, in units of 10 laps."""
    age = d.tyre_age.to_numpy(float) / 10
    ref = (d.stint_start_age.to_numpy(float) + (REF_LAPS - 1) / 2) / 10
    return age - ref, age ** 2 - ref ** 2


def curve_rows(cond, ages, start_age=1, stint_number=1, start_lap=1):
    """Feature rows for one stint on fresh tyres, under fixed race conditions (a dict or Series)."""
    ages = np.asarray(ages)
    rows = pd.DataFrame({c: cond[c] for c in COND if c not in ("stint_number", "stint_start_frac")},
                        index=range(len(ages)))
    rows["stint_number"] = stint_number
    rows["stint_start_frac"] = start_lap / cond["total_laps"]
    rows["tyre_age"] = ages
    rows["stint_start_age"] = start_age
    rows["stint_lap"] = ages - start_age + 1
    rows["in_traffic"] = 0
    return add_wear(rows)


def add_wear(d):
    d["wear_lin"], d["wear_quad"] = wear_terms(d)
    return d
