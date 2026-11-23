"""Train and test every model under two protocols.

  cv        5-fold cross-validation grouped by circuit: each test circuit is one the model
            has never seen in either season ("can it handle a new track?").
  temporal  train on 2023, test on 2024 ("can it predict next season?").

Out-of-fold predictions go to data/preds_<protocol>.csv, fitted models to
models/<protocol>/<name>_<fold>.joblib, and the chosen hyperparameters to data/tuning.json.

    python -m f1sim.evaluate                       # all models, both protocols
    python -m f1sim.evaluate cv rf gbm             # one protocol, some models
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

from f1sim.common import N_FOLDS
from f1sim.models import MODELS

PROTOCOLS = ["cv", "temporal"]


def splits(d, protocol):
    if protocol == "cv":
        return [(str(k), d.fold != k, d.fold == k) for k in range(N_FOLDS)]
    return [("2024", d.Year == 2023, d.Year == 2024)]


def run(protocol, names):
    d = pd.read_csv("data/laps.csv")
    path = f"data/preds_{protocol}.csv"
    preds = pd.read_csv(path) if os.path.exists(path) else d[["RaceId", "Driver", "LapNumber"]].copy()
    tuning = json.load(open("data/tuning.json")) if os.path.exists("data/tuning.json") else {}
    os.makedirs(f"models/{protocol}", exist_ok=True)
    for name in names:
        t0, p = time.time(), np.full(len(d), np.nan)
        for fold, tr, te in splits(d, protocol):
            m = MODELS[name][1]().fit(d[tr], d.deg[tr], groups=d.CircuitKey[tr])
            p[te.to_numpy()] = m.predict(d[te])
            joblib.dump(m.model, f"models/{protocol}/{name}_{fold}.joblib", compress=3)
            tuning.setdefault(protocol, {}).setdefault(name, {})[fold] = {k: str(v) for k, v in m.best.items()}
        preds[name] = p
        mask = ~np.isnan(p)
        mae = np.abs(p[mask] - d.deg[mask]).mean()
        print(f"  {protocol:<9}{name:<10}lap MAE {mae:.3f} s   ({time.time() - t0:.0f} s)", flush=True)
        preds.to_csv(path, index=False)
        json.dump(tuning, open("data/tuning.json", "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    protocols = [a for a in args if a in PROTOCOLS] or PROTOCOLS
    names = [a for a in args if a in MODELS] or list(MODELS)
    for proto in protocols:
        run(proto, names)
