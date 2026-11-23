"""Correctness tests.   python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from f1sim import strategy as S
from f1sim.nn import MLP, DegNet, _toy, gradient_check


@pytest.mark.parametrize("cls", [MLP, DegNet])
def test_gradient_check(cls):
    d, y = _toy(64)
    assert gradient_check(cls(hidden=(8, 4), wd=0.0), d, y) < 1e-5


@pytest.mark.parametrize("cls", [MLP, DegNet])
def test_overfit_small_batch(cls):
    d, y = _toy(64)
    net = cls(hidden=(32, 16), lr=3e-3, wd=0.0, epochs=1500, batch=64, patience=10 ** 9, val_frac=0).fit(d, y)
    assert net.history[-1][0] < 0.05


def test_degnet_never_predicts_faster_tyres():
    d, y = _toy(256)
    net = DegNet(hidden=(8, 4), epochs=20).fit(d, y)
    rate, curv = net.curve_params(d)
    assert (rate >= 0).all() and (curv >= 0).all()


def test_folds_hold_out_whole_circuits():
    d = pd.read_csv("data/laps.csv", usecols=["CircuitKey", "fold"])
    assert d.groupby("CircuitKey").fold.nunique().max() == 1


def test_no_test_race_leaks_into_its_model():
    """Every race is planned by the fold model that never saw that race's circuit."""
    r = S.races()
    d = pd.read_csv("data/laps.csv", usecols=["RaceId", "CircuitKey", "fold"])
    for rid, race in r[r.used].iterrows():
        k = int(race.fold)
        train_circuits = set(d[d.fold != k].CircuitKey)
        assert race.CircuitKey not in train_circuits


def test_parse_strategy():
    assert S.parse_strategy("M-H:22") == (("medium", "hard"), (22,))
    assert S.parse_strategy("s-m-h:15,40") == (("soft", "medium", "hard"), (15, 40))
    with pytest.raises(ValueError):
        S.parse_strategy("M-H:10,20")


def test_simulation_adds_up():
    r = S.races()
    rid = r[r.comparable].index[0]
    sc = S.StintCosts(S.load_model("baseline", rid), rid)
    laps = S.simulate(rid, ("medium", "hard"), (20,), sc)
    assert len(laps) == int(r.loc[rid].total_laps)
    assert np.isclose(sum(x["lap_time"] for x in laps), laps[-1]["total"])
    assert sum(x["pit"] > 0 for x in laps) == 1
    # a stop costs exactly the race's pit loss
    assert np.isclose(sum(x["pit"] for x in laps), r.loc[rid].pit_loss)


def test_optimiser_picks_minimum():
    r = S.races()
    rid = r[r.comparable].index[0]
    sc = S.StintCosts(S.load_model("baseline", rid), rid)
    ps, cost = S.one_stop(sc, "medium", "hard", r.loc[rid].pit_loss)
    best, lo, hi = S.pit_window(sc, "medium", "hard", r.loc[rid].pit_loss)
    assert best == ps[np.argmin(cost)] and lo <= best <= hi
